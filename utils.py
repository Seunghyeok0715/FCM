import torch
import argparse
import numpy as np
import math
import ast
from pytorch3d.transforms import euler_angles_to_matrix
from RepKPU.models.utils import *

def str2bool(val):
    if isinstance(val, bool):
        return val
    if val.lower() in ['yes', 'true', 't', 'y']:
        return True
    elif val.lower() in ['no', 'false', 'f', 'n']:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def reset_model_args(args1, args2):
    for arg in vars(args1):
        setattr(args2, arg, getattr(args1, arg))

def DPS(A, b, x, scale):
    with torch.enable_grad():
        x = x.detach().requires_grad_()
        norm = x.new_zeros(1)
        for i in range(len(A)):
            Ax, pc = A[i].forward(x)
            r = (b[i] - Ax) 
            if i == 0:
                loss = torch.linalg.norm(r)/(len(A))
            else:
                loss += torch.linalg.norm(r)/(len(A))
        loss.backward()
        d_x = x.grad
        d_x[:,3:,:] *= 75
        x = x - d_x * scale 
    return x

def loss_func(A,b,x):
    for i in range(len(A)):
        Ax, pc = A[i].forward(x)
        r = (b[i] - Ax)
        if i == 0:
            loss = torch.linalg.norm(r)/(len(A))
        else:
            loss += torch.linalg.norm(r)/(len(A))
    return loss 

def FCM(A, b, x, lr, eta=1e-4, L=1, eps=1e-12):
    with torch.enable_grad():
        x = x.detach().requires_grad_()
        
        loss = loss_func(A,b,x)
        loss.backward()
        d_x = x.grad
        d_x[:,3:,:] *= (d_x[:,:3,:].std()/d_x[:,3:,:].std())
        
        g = d_x.clone()

        beta = (torch.norm(x) / torch.norm(g)) * lr
        x = x.detach()
        
        x_temp = x - beta * g
        x_temp = x_temp.detach().requires_grad_()
        loss_temp = loss_func(A,b,x_temp)
        loss_temp.backward()
        d_x_temp = x_temp.grad
        d_x_temp[:,3:,:] *= (d_x_temp[:,:3,:].std()/d_x_temp[:,3:,:].std())
        g_temp = d_x_temp.clone()
        
        h = (g - g_temp)/beta
        
        alpha = (torch.norm(g)**2)/((torch.dot(g.reshape(-1), h.reshape(-1))+eps))
        
        alpha = min(alpha, 1.5/L)
        x_hat = x - alpha * g
        loss_hat = loss_func(A,b,x_hat)
        if loss_hat > loss - eta * alpha * torch.norm(g)**2:
            alpha *= 0.5
            x_hat = x - alpha * g
    return x_hat

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def get_beta_schedule(beta_schedule, *, num_diffusion_timesteps, beta_start=0.02, beta_end=0.0001):
    if beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
        
    elif beta_schedule == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        )
        
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def get_sigma(t, sde):
    sigma_t = sde.sigma_min * (sde.sigma_max / sde.sigma_min) ** t
    return sigma_t


def pred_x0_from_s(xt, s, t, sde):
    sigma_t = get_sigma(t, sde)
    tmp = sigma_t.view(sigma_t.shape[0], 1, 1, 1)
    pred_x0 = xt + (tmp ** 2) * s
    return pred_x0


def recover_xt_from_x0(x0_t, s, t, sde):
    sigma_t = get_sigma(t, sde)
    tmp = sigma_t.view(sigma_t.shape[0], 1, 1, 1)
    xt = x0_t - (tmp ** 2) * s
    return xt


def pred_eps_from_s(s, t, sde):
    sigma_t = get_sigma(t, sde)
    tmp = sigma_t.view(sigma_t.shape[0], 1, 1, 1)
    pred_eps = -tmp * s
    return pred_eps

def get_RTF(metadata, i, device):
    with open(metadata, 'r') as file:
        data = file.read()
    data = data.replace('\n', '')
    data_list = ast.literal_eval(data)
    data_array = np.array(data_list)
    array = data_array[i]

    yaw_deg = array[0]
    pitch_deg = array[1]
    roll_deg = array[2]
    dist_ratio = array[3]
    focal_len = array[4]
    sensor_size = array[5]
    max_dist = array[6]
    
    roll_rad = torch.deg2rad(torch.tensor(roll_deg))
    pitch_rad = torch.deg2rad(torch.tensor(90-pitch_deg))
    yaw_rad = torch.deg2rad(torch.tensor(yaw_deg))

    R_matrix = euler_angles_to_matrix(torch.tensor([roll_rad, pitch_rad, yaw_rad]), "ZXY")
    distance = dist_ratio * max_dist
    camera_position = np.array([0, 0, distance])
    camera_position_world = R_matrix @ camera_position
    T_vector = -camera_position_world
    
    
    fov = 2 * math.degrees(math.atan(sensor_size / (2 * focal_len)))
    
    to_p3d =torch.tensor([[1, 0, 0, 0],
                          [0, 0, 1, 0],
                          [0, -1, 0, 0],
                          [0, 0, 0, 1]],dtype=torch.float32, device=device)
    
    Rt = torch.zeros((4,4),dtype=torch.float32,device=device)
    Rt[:3,:3]=R_matrix
    Rt[:3,3]=T_vector
    Rt = to_p3d@Rt
    R = Rt[:3,:3]
    T = Rt[:3,3]
    R_inv = torch.transpose(R, 0, 1)
    T_cam = -torch.matmul(R_inv, T)
    
    return R_inv,T_cam,fov
    
def fscore(dist1, dist2, threshold=0.001):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2
