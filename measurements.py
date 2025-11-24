'''This module handles task-dependent operations (A) and noises (n) to simulate a measurement y=Ax+n.'''
import os
from abc import ABC
from torchvision import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
)
import matplotlib.pyplot as plt
from typing import cast


# =================
# Operation classes
# =================

__OPERATOR__ = {}

def register_operator(name: str):
    def wrapper(cls):
        if __OPERATOR__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __OPERATOR__[name] = cls
        return cls
    return wrapper


def get_operator(name: str, **kwargs):
    if __OPERATOR__.get(name, None) is None:
        raise NameError(f"Name {name} is not defined.")
    return __OPERATOR__[name](**kwargs)


def signed_clamp(x, eps):
    sign = x.sign() + (x == 0.0).type_as(x)
    x_clamp = sign * torch.clamp(x.abs(), eps)
    return x_clamp

class RenderOperator(ABC):
    def __init__(self, R, T, fov, mean, std, radius, device):
        self.device = device
        self.bg_color = None
        self.cameras = FoVPerspectiveCameras(device=device, R=R, T=T, fov=fov, znear=0.01)
        camera_trivial = self.cameras.clone()
        camera_trivial.R[:] = torch.eye(3)
        camera_trivial.T *= 0.0
        
        raster_settings = PointsRasterizationSettings(image_size=224, 
                                                      radius = radius,
                                                      points_per_pixel = 10
                                                      )
        self.rasterizer = PointsRasterizer(cameras=camera_trivial, raster_settings=raster_settings)
        self.r= self.rasterizer.raster_settings.radius
        
        self.std = std
        self.mean = mean
    
    def output_to_point_clouds(self, output: torch.Tensor) -> Pointclouds:
        xyz, rgb = output[:,:3,:], output[:,3:,:]
        unscale_xyz = (xyz + 1)/2
        unscale_rgb = (rgb + 1)/2
        pc = Pointclouds(
                points=unscale_xyz.permute(0,2,1) * self.std[:3] + self.mean[:3],
                features=unscale_rgb.permute(0,2,1) * self.std[3:] + self.mean[3:]
                )
        return pc

    def forward(self, data):
        pc = self.output_to_point_clouds(data)
        pts_world = pc.points_padded()
        pts_view = self.cameras.get_world_to_view_transform().transform_points(pts_world ,eps = 1e-2)
        # it is crucial to actually clamp the points as well ...
        pts_view = torch.cat(
        (pts_view[..., :-1], signed_clamp(pts_view[..., -1:], 1e-2)), dim=-1
        )
        pc = pc.update_padded(pts_view)
        
        featdim = pc.features_packed().shape[-1]
        fragments = self.rasterizer(pc)
        dists2 = fragments.dists
        weights = 1 - dists2 / (self.r ** 2)
        ok = cast(torch.BoolTensor, (fragments.idx >= 0)).float()
        
        weights = weights*ok
        
        fragments_prm = fragments.idx.long().permute(0, 3, 1, 2)
        weights_prm = weights.permute(0, 3, 1, 2)
        render_img = AlphaCompositor()(
            fragments_prm,
            weights_prm,
            pc.features_packed().permute(1, 0),
            background_color=self.bg_color if self.bg_color is not None else [0.0] * featdim
        )
        render_mask = -torch.prod(1.0 - weights, dim=-1) + 1.0
        
        return torch.concat((render_img, render_mask.unsqueeze(1)),dim=1).permute(0,2,3,1) , pc
    
    def forward_eval(self,data):
        pc = self.output_to_point_clouds(data)
        pc_ = pc.clone()
        pts_world = pc.points_padded()
        pts_view = self.cameras.get_world_to_view_transform().transform_points(pts_world ,eps = 1e-2)
        # it is crucial to actually clamp the points as well ...
        pts_view = torch.cat(
        (pts_view[..., :-1], signed_clamp(pts_view[..., -1:], 1e-2)), dim=-1
        )
        pc = pc.update_padded(pts_view)
        
        featdim = pc.features_packed().shape[-1]
            
        fragment = self.rasterizer(pc)
        dists2 = fragment.dists
        weights = 1 - dists2 / (self.r ** 2)
        ok = cast(torch.BoolTensor, (fragment.idx >= 0)).float()

        weights = weights*ok

        fragments_prm = fragment.idx.long().permute(0, 3, 1, 2)
        weights_prm = weights.permute(0, 3, 1, 2)
        render_img = AlphaCompositor()(
            fragments_prm,
            weights_prm,
            pc.features_packed().permute(1, 0),
            background_color=self.bg_color if self.bg_color is not None else [0.0] * featdim
        )

        render_mask = -torch.prod(1.0 - weights, dim=-1) + 1.0
        return torch.concat((render_img, render_mask.unsqueeze(1)),dim=1).permute(0,2,3,1) , pc_