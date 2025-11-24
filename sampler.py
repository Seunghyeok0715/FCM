import os
import tqdm
import yaml
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from measurements import RenderOperator
from utils import FCM, get_beta_schedule, get_RTF, fscore
from plyfile import PlyData,PlyElement
from RepKPU.models.utils import *
from pytorch3d.loss import chamfer_distance
from emd_utils.emd_module import emdModule

def get_num_list(n_views):
    if n_views <= 0:
        return []
    step = 36 // n_views
    return [i * step for i in range(n_views)]

def save_ply(output, path):
    output = output.detach().cpu().numpy()
    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z']]
    elements = np.empty(output.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, output))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    
def save_ply_color(output, path):
    output = output.detach().cpu().numpy()
    dtype_full = [(attribute, 'f4') for attribute in ['x', 'y', 'z','r','g','b']]
    elements = np.empty(output.shape[0], dtype=dtype_full)
    elements[:] = list(map(tuple, output))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
   
def upsampling(args, model, input_pcd):
    pcd_pts_num = input_pcd.shape[-1]
    patch_pts_num = args.num_points
    sample_num = int(pcd_pts_num / patch_pts_num * args.patch_rate)
    seed = FPS(input_pcd, sample_num)
    patches = extract_knn_patch(patch_pts_num, input_pcd, seed)
    patches, centroid, furthest_distance = normalize_point_cloud(patches)
    coarse_pts, _= model.forward(patches)
    coarse_pts = centroid + coarse_pts * furthest_distance
    coarse_pts = rearrange(coarse_pts, 'b c n -> c (b n)').contiguous()
    coarse_pts = FPS(coarse_pts.unsqueeze(0), input_pcd.shape[-1]* args.up_rate)
    return coarse_pts

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.device = device
        self.config = config
        self.model_var_type = config.model_var_type
        betas = get_beta_schedule(
            beta_schedule=config.noise_schedule,
            num_diffusion_timesteps=config.steps,
        )
        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0 - betas
        alphas_cumprod = alphas.cumprod(dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(device), alphas_cumprod[:-1]], dim=0
        )
        self.alphas_cumprod_prev = alphas_cumprod_prev
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        if self.model_var_type == "fixedlarge":
            self.logvar = betas.log()
        elif self.model_var_type == "fixedsmall":
            self.logvar = posterior_variance.clamp(min=1e-20).log()
            
    def fcm_sampling(self, model,upsample_model):
        args, config = self.args, self.config
        
        class_dict = {
                  "airplane": "02691156",
                  "car": "02958343"
                  }
    
        class_n = class_dict.get(args.object, "Unknown class")
        
        data_array = []
        frame_array = []
        
        with open(args.data_lists, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    data, frame = line.split('/')
                    data_array.append(data)
                    frame_array.append(int(frame))
        
        if not len(data_array) == len(frame_array):
            raise ValueError("Mismatch between data_array and frame_array lengths.")
        
        batch_size = args.batch_size
        num_batches = (len(data_array) + batch_size - 1) // batch_size
        
        out_path = args.save_dir
        os.makedirs(out_path, exist_ok=True)
        for img_dir in ['recon', 'label', 'progress']:
            os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)
            
        for recon_dir in ['img','color','xyz']:
            os.makedirs(os.path.join(out_path, 'recon', recon_dir), exist_ok=True)
            
        # mask_blur = GaussianBlur(kernel_size=5, sigma=1.0)
        
        class_data = load_yaml(args.data_config)
        
        mean = torch.tensor(class_data[args.object]['mean'],device=self.device)
        std = torch.tensor(class_data[args.object]['std'],device=self.device)
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(data_array))
            batch_data = data_array[start_idx:end_idx]
    
            if args.N_view == 1:
                # Load images & cameras in batch
                batch_frames = frame_array[start_idx:end_idx]
                
                gt_xyz,sample_xyz =[],[]
                ref_imgs,Rs, Ts, fovs = [], [], [], []
                for i, idx in enumerate(batch_data):
                    # Load image & camera
                    img_dir = os.path.join(args.data_dir,class_n,idx,"easy","{:02d}.png".format(int(batch_frames[i])))
                    ref_img_ = Image.open(img_dir).convert('RGBA')
                    ref_img = np.array(ref_img_)
                    ref_img = ref_img / 255.0
                    ref_img = torch.tensor(ref_img,dtype=torch.float32,device=self.device)
                    ref_imgs.append(ref_img)
                    plt.imsave(os.path.join(out_path, 'label', f'ref_img{start_idx + i:02d}.png'),
                               torch.clip(ref_img,0,1).detach().cpu().numpy())

                    metadata_dir = os.path.join(args.data_dir,class_n,idx,"easy/rendering_metadata.txt")
                    R,T,fov = get_RTF(metadata_dir,int(batch_frames[i]),device=self.device)
                    Rs.append(R)
                    Ts.append(T)
                    fovs.append(fov)

                    gt_xyz_ = np.load(os.path.join(args.gt_dir,class_n,f'{idx}.npy'))[:8192,:]
                    gt_xyz_ = torch.tensor(gt_xyz_,device=self.device)

                    gt_xyz_ = gt_xyz_ - (gt_xyz_.max(dim=0)[0]+gt_xyz_.min(dim=0)[0])/2
                    gt_xyz.append(gt_xyz_.detach().cpu())
                    
                ref_imgs = torch.stack(ref_imgs)
                
                Rs = torch.stack(Rs)
                Ts = torch.stack(Ts)
                fovs = torch.tensor(fovs,device=self.device)

                class_data = load_yaml(args.data_config)
                mean = torch.tensor(class_data[args.object]['mean'],device=self.device)
                std = torch.tensor(class_data[args.object]['std'],device=self.device)
                
                operators = [RenderOperator(R=Rs, T=Ts, fov=fovs, mean=mean, std=std, radius = args.radius,device=self.device)]

                y_ns = [ref_imgs]
                
            else:
                # Load images & cameras in batch
                view_list = get_num_list(args.N_view)
                gt_xyz,sample_xyz, y_ns,operators = [],[],[],[]
                for j, view_num in enumerate(view_list):
                    
                    b_ref_img, b_R, b_T, b_fov = [], [], [], []
                    for i, idx in enumerate(batch_data):
                        # Load image & camera
                        img_dir = os.path.join(args.data_dir,class_n,idx,"easy","{:02d}.png".format(view_num))
                        ref_img_ = Image.open(img_dir).convert('RGBA')
                        ref_img = np.array(ref_img_)
                        ref_img = ref_img / 255.0
                        ref_img = torch.tensor(ref_img,dtype=torch.float32,device=self.device)
                        b_ref_img.append(ref_img)

                        metadata_dir = os.path.join(args.data_dir,class_n,idx,"easy/rendering_metadata.txt")
                        R,T,fov = get_RTF(metadata_dir,view_num,device=self.device)
                        b_R.append(R)
                        b_T.append(T)
                        b_fov.append(fov)

                        if j == 0:
                            plt.imsave(os.path.join(out_path, 'label', f'ref_img{start_idx + i:02d}.png'),
                                       torch.clip(ref_img,0,1).detach().cpu().numpy())
                            gt_xyz_ = np.load(os.path.join(args.gt_dir,class_n,f'{idx}.npy'))[:8192,:]
                            gt_xyz_ = torch.tensor(gt_xyz_,device=self.device)

                            gt_xyz_ = gt_xyz_ - (gt_xyz_.max(dim=0)[0]+gt_xyz_.min(dim=0)[0])/2
                            gt_xyz.append(gt_xyz_.detach().cpu())

                    b_ref_img = torch.stack(b_ref_img)
                    b_R = torch.stack(b_R)
                    b_T = torch.stack(b_T)
                    b_fov = torch.tensor(b_fov,device=self.device)
                    b_y_n = b_ref_img

                    operator = RenderOperator(R=b_R, T=b_T, fov=b_fov, mean=mean, std=std, radius = args.radius,device=self.device)
                    operators.append(operator)
                    y_ns.append(b_y_n)
        
            x = torch.randn((len(batch_data),6,2048), device=self.device)

            with torch.no_grad():
                skip = config.steps//args.T_sampling
                n = x.size(0)
                x0_preds = []
                xs = [x]

                # generate time schedule
                times = range(0, 1024, skip)
                times_next = [-1] + list(times[:-1])
                times_pair = zip(reversed(times), reversed(times_next))

                # reverse diffusion sampling
                for i, j in tqdm.tqdm(times_pair, total=len(times)):
                    t = (torch.ones(n) * i).to(x.device)
                    next_t = (torch.ones(n) * j).to(x.device)

                    at = compute_alpha(self.betas, t.long())
                    at_next = compute_alpha(self.betas, next_t.long())

                    xt = xs[-1].to(self.device)

                    # 0. NFE
                    et = model(xt, t)
                    et = et[:, :et.size(1)//2]

                    # 1. Tweedie
                    x0_t = (xt - et * (1 - at).sqrt()) / at.sqrt()
                    x0_t = torch.clip(x0_t,-1.0,1.0)
                    
                    # 2. FCM
                    for k in range(4):
                        if k == 0:
                            x0_t_hat = FCM(operators, y_ns, x0_t, lr = args.alpha_0, L=args.num_L)
                        else:
                            x0_t_hat = FCM(operators, y_ns, x0_t_hat, lr = args.alpha_0, L=args.num_L)
                            
                    eta = self.args.eta

                    c1 = ((1 - at/at_next)*(1-at_next)/(1-at)).sqrt() * eta
                    c2 = ((1 - at_next)-c1 ** 2).sqrt()

                    # DDIM sampling
                    if j != 0:
                        xt_next = at_next.sqrt() * x0_t_hat + c1 * torch.randn_like(x0_t_hat) + c2 * et
                        
                    # Final step
                    else:
                        xt_next = x0_t_hat
                        
                    x0_preds.append(x0_t.to('cpu'))
                    xs.append(xt_next.to('cpu'))
                    
                x = xs[-1].to(device=self.device)
                
                img, pc = operators[0].forward_eval(x)
                sample = (x + 1)/2
                sample = sample * std.unsqueeze(0).unsqueeze(-1).repeat(len(batch_data), 1, sample.shape[2]) + mean.unsqueeze(0).unsqueeze(-1).repeat(len(batch_data), 1, sample.shape[2])
                sample_xyz_ = sample[:,:3,:]
                sample = sample.permute(0,2,1)
                # upsampling
                input_pcd, centroid, furthest_distance = normalize_point_cloud(sample_xyz_)
                for i in range(len(batch_data)):            
                    xyz_upsampled = upsampling(args, upsample_model, input_pcd[i].unsqueeze(0))
                    xyz_upsampled = centroid[i].unsqueeze(0) + xyz_upsampled * furthest_distance[i].unsqueeze(0)
                    xyz_upsampled=xyz_upsampled.permute(0,2,1).squeeze()
                    sample_xyz.append(xyz_upsampled.detach().cpu())
                    save_ply_color(sample[i], os.path.join(out_path, 'recon', 'color',f'sample{start_idx + i:02d}.ply'))
                    save_ply(xyz_upsampled, os.path.join(out_path, 'recon', 'xyz',f'sample{start_idx + i:02d}.ply'))
                    plt.imsave(os.path.join(out_path, 'recon', 'img',f'sample{start_idx + i:02d}.png'), torch.clip(img[i],0,1).contiguous().detach().cpu().numpy())

                # evaluate
                sample_xyz = torch.stack(sample_xyz).to(device=self.device)
                gt_xyz = torch.stack(gt_xyz).to(device=self.device)

                chamLoss = chamfer_3DDist()

                dist1, dist2, _, _ = chamLoss(sample_xyz, gt_xyz)

                f_s, _, _ = fscore(torch.sqrt(dist1), torch.sqrt(dist2), 0.01)
                cd_l2_s = (torch.mean(dist1,dim = 1)) + (torch.mean(dist2,dim = 1))
                cd_l1_s,_ = chamfer_distance(sample_xyz, gt_xyz, norm=1)
                    
                EMD = emdModule()
                emd_s, _ = EMD(sample_xyz, gt_xyz, 0.002, 10000)
                emd_s = torch.sqrt(emd_s)
                
                if batch_idx == 0:
                    f_score = f_s.mean().detach().cpu().unsqueeze(0)
                    cd_l1_score = cd_l1_s.mean().detach().cpu().unsqueeze(0)
                    cd_l2_score = cd_l2_s.mean().detach().cpu().unsqueeze(0)
                    emd_score = emd_s.mean().detach().cpu().unsqueeze(0)
                else:
                    f_score = torch.cat([f_score, f_s.mean().detach().cpu().unsqueeze(0)])
                    cd_l1_score = torch.cat([cd_l1_score, cd_l1_s.mean().detach().cpu().unsqueeze(0)])
                    cd_l2_score = torch.cat([cd_l2_score, cd_l2_s.mean().detach().cpu().unsqueeze(0)])
                    emd_score = torch.cat([emd_score, emd_s.mean().detach().cpu().unsqueeze(0)])
                    
                print(f'-----{batch_idx+1}/{num_batches}-----')
                print(f'f_score:{f_s.mean().item()}/{f_score.mean().item()}')
                print(f'cd_l1_score:{cd_l1_s.mean().item()}/{cd_l1_score.mean().item()}')
                print(f'cd_l2_score:{cd_l2_s.mean().item()}/{cd_l2_score.mean().item()}')
                print(f'emd_score:{emd_s.mean().item()}/{emd_score.mean().item()}')
                print('---------------------------------')

        print(f'f_score:{f_score.mean().item()}')
        print(f'cd_l1_score:{cd_l1_score.mean().item()}')
        print(f'cd_l2_score:{cd_l2_score.mean().item()}')
        print(f'emd_score:{emd_score.mean().item()}')
        
def compute_alpha(beta, t):
    beta = torch.cat([torch.zeros(1).to(beta.device), beta], dim=0)
    a = (1 - beta).cumprod(dim=0).index_select(0, t + 1).view(-1, 1, 1)
    return a
