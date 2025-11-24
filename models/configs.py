from typing import Any, Dict

import torch
import torch.nn as nn

from .sdf import CrossAttentionPointCloudSDFModel
from .transformer import (
    CLIPImageGridPointDiffusionTransformer,
    CLIPImageGridUpsamplePointDiffusionTransformer,
    CLIPImagePointDiffusionTransformer,
    PointDiffusionTransformer,
    UpsamplePointDiffusionTransformer,
)

MODEL_CONFIGS = {
    "8192_color": {
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 6,
        "layers": 12,
        "n_ctx": 8192,
        "name": "PointDiffusionTransformer",
        "output_channels": 12,
        "time_token_cond": True,
        "width": 512,
    },
    "2048_color": {
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 6,
        "layers": 12,
        "n_ctx": 2048,
        "name": "PointDiffusionTransformer",
        "output_channels": 12,
        "time_token_cond": True,
        "width": 512,
    },
   
    "8192_xyz": {
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 3,
        "layers": 12,
        "n_ctx": 8192,
        "name": "PointDiffusionTransformer",
        "output_channels": 6,
        "time_token_cond": True,
        "width": 512,
    },
    "2048_xyz": {
        "heads": 8,
        "init_scale": 0.25,
        "input_channels": 3,
        "layers": 12,
        "n_ctx": 2048,
        "name": "PointDiffusionTransformer",
        "output_channels": 6,
        "time_token_cond": True,
        "width": 512,
    },
}


def model_from_config(config: Dict[str, Any], device: torch.device) -> nn.Module:
    config = config.copy()
    name = config.pop("name")
    if name == "PointDiffusionTransformer":
        return PointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "CLIPImagePointDiffusionTransformer":
        return CLIPImagePointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "CLIPImageGridPointDiffusionTransformer":
        return CLIPImageGridPointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "UpsamplePointDiffusionTransformer":
        return UpsamplePointDiffusionTransformer(device=device, dtype=torch.float32, **config)
    elif name == "CLIPImageGridUpsamplePointDiffusionTransformer":
        return CLIPImageGridUpsamplePointDiffusionTransformer(
            device=device, dtype=torch.float32, **config
        )
    elif name == "CrossAttentionPointCloudSDFModel":
        return CrossAttentionPointCloudSDFModel(device=device, dtype=torch.float32, **config)
    raise ValueError(f"unknown model name: {name}")
