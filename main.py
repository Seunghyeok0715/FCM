import sys
import argparse
import logging
import yaml
import torch
from models.configs import MODEL_CONFIGS, model_from_config

from RepKPU.models.repkpu import RepKPU_o
from configs.args_parser import args_parser
from RepKPU.models.utils import *

from sampler import Diffusion

torch.set_printoptions(sci_mode=False)

def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])

    parser.add_argument("--diffusion_config",default='./configs/diffusion_config.yaml',type=str, help="Path to the config file")
    parser.add_argument("--model", type=str, default='/path/to/airplane.pth', help="Load pre-trained ckpt")
    parser.add_argument("--eta", type=float, default=1.0, help="Eta")
    parser.add_argument("--T_sampling", type=int, default=256, help="Total number of sampling steps")
    parser.add_argument('--object', default='airplane')
    parser.add_argument('--data_dir', default='/path/to/albedo/test')
    parser.add_argument('--data_lists', default='./data_lists/matched_files_airplane.txt')
    parser.add_argument('--gt_dir', default='/path/to/test_ply')
    parser.add_argument('--data_config', type=str, default='./configs/class_data.yaml')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--radius', type=float, default=0.018) # car: 0.027
    parser.add_argument('--N_view', type=int, default=1) 
    parser.add_argument('--alpha_0', type=float, default=0.02)
    parser.add_argument('--num_L', default=1.0, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    
    ## upsample args ##
    parser = args_parser(parser)
    args = parser.parse_args()

    # parse config file
    with open(args.diffusion_config, "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    # add device
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("Using device: {}".format(device))
    new_config.device = device

    torch.backends.cudnn.benchmark = True

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def main():
    
    args, config = parse_args_and_config()
    
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_str)  
    torch.cuda.set_device(device_str)
    
    sampler = Diffusion(args, config,device)
    model = model_from_config(MODEL_CONFIGS['2048_color'], device).to("cuda")
    upsample_model = RepKPU_o(args).to(device)          
    ckpt = torch.load(args.model, map_location=device)
    
    model.load_state_dict(ckpt['model_state'])
    upsample_model.load_state_dict(torch.load(args.upsample_model, map_location=device))
    
    model.eval()
    upsample_model.eval()
    
    sampler.fcm_sampling(model, upsample_model)    
    return 0


if __name__ == "__main__":
    sys.exit(main())
