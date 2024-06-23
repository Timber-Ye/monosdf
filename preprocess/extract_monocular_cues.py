# adapted from https://github.com/EPFL-VILAB/omnidata
import torch
import torch.nn.functional as F
from torchvision import transforms

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import argparse
import os.path
from pathlib import Path
import glob
import sys

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn, TimeRemainingColumn
from nerfstudio.utils.rich_utils import ItersPerSecColumn


parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

parser.add_argument('--omnidata_path', dest='omnidata_path', help="path to omnidata model")
parser.set_defaults(omnidata_path='/home/yuzh/Projects/omnidata/omnidata_tools/torch/')

parser.add_argument('--pretrained_models', dest='pretrained_models', help="path to pretrained models")
parser.set_defaults(pretrained_models='/home/yuzh/Projects/omnidata/omnidata_tools/torch/pretrained_models/')

parser.add_argument('--task', dest='task', help="normal or depth")
parser.set_defaults(task='NONE')

parser.add_argument('--img_path', dest='img_path', help="path to rgb image")
parser.set_defaults(im_name='NONE')

parser.add_argument('--scene_id', dest='scene_id', help="scene id")
parser.set_defaults(scene_id='scene0488_01')

parser.add_argument('--output_path', dest='output_path', help="path to where output image should be stored")
parser.set_defaults(store_name='NONE')

args = parser.parse_args()

root_dir = args.pretrained_models 
omnidata_path = args.omnidata_path

sys.path.append(args.omnidata_path)
print(sys.path)
from omni_modules.unet import UNet
from omni_modules.midas.dpt_depth import DPTDepthModel
from omni_data.transforms import get_transform

trans_topil = transforms.ToPILImage()
os.system(f"mkdir -p {args.output_path}")
map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

H, W = 480, 640

# get target task and model
if args.task == 'normal':
    image_size = 384
    
    ## Version 1 model
    # pretrained_weights_path = root_dir + 'omnidata_unet_normal_v1.pth'
    # model = UNet(in_channels=3, out_channels=3)
    # checkpoint = torch.load(pretrained_weights_path, map_location=map_location)

    # if 'state_dict' in checkpoint:
    #     state_dict = {}
    #     for k, v in checkpoint['state_dict'].items():
    #         state_dict[k.replace('model.', '')] = v
    # else:
    #     state_dict = checkpoint
    
    
    pretrained_weights_path = root_dir + 'omnidata_dpt_normal_v2.ckpt'
    model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        get_transform('rgb', image_size=None)])

elif args.task == 'depth':
    image_size = 384
    pretrained_weights_path = root_dir + 'omnidata_dpt_depth_v2.ckpt'  # 'omnidata_dpt_depth_v1.ckpt'
    # model = DPTDepthModel(backbone='vitl16_384') # DPT Large
    model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
    checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
    if 'state_dict' in checkpoint:
        state_dict = {}
        for k, v in checkpoint['state_dict'].items():
            state_dict[k[6:]] = v
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.to(device)
    trans_totensor = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                        transforms.CenterCrop(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=0.5, std=0.5)])

else:
    print("task should be one of the following: normal, depth")
    sys.exit()

trans_rgb = transforms.Compose([transforms.Resize(image_size, interpolation=PIL.Image.BILINEAR),
                                transforms.CenterCrop(image_size),
                                ])


def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img


def save_outputs(img_path, output_file_name):
    with torch.no_grad():
        save_path = os.path.join(args.output_path, f'{output_file_name}.png')
        rgb_path = os.path.join(args.output_path, '..', 'rgb', f'{output_file_name}.png')

        if os.path.exists(save_path):
            return

        # print(f'Reading input {img_path} ...')
        img = Image.open(img_path)
        trans_rgb(img).save(rgb_path)
        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        if args.task == 'depth':
            #output = F.interpolate(output.unsqueeze(0), (512, 512), mode='bicubic').squeeze(0)
            output = output.clamp(0,1)
            
            np.save(save_path.replace('.png', '.npy'), output.detach().cpu().numpy().astype(np.float16)[0])
            
            #output = 1 - output
#             output = standardize_depth_map(output)
            plt.imsave(save_path, output.detach().cpu().squeeze(),cmap='viridis')
            
        else:
            #import pdb; pdb.set_trace()
            normal = output.detach().cpu().numpy()[0]
            np.save(save_path.replace('.png', '.npy'), normal.astype(np.float16))
            trans_topil(output[0]).save(save_path)
            
        # print(f'Writing output {save_path} ...')

img_path = os.path.join(args.img_path, args.scene_id, 'color')
pose_path = os.path.join(args.img_path, args.scene_id, 'pose')
intrinsic_path = os.path.join(args.img_path, args.scene_id, 'intrinsic')
args.output_path = os.path.join(args.output_path, args.scene_id, 'MonoSDF', 'omnidata', args.task)

if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

if not os.path.exists(os.path.join(args.output_path, '..', 'rgb')):
    os.makedirs(os.path.join(args.output_path, '..', 'rgb'))

if Path(img_path).is_file():
    save_outputs(args.img_path, os.path.splitext(os.path.basename(args.img_path))[0])
elif Path(img_path).is_dir():
    progress = Progress(
        TextColumn(f":ten_o’clock: Predicting monocular {args.task} :ten_o’clock:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )

    img_file_list = sorted(glob.glob(img_path+'/*'))
    pose_file_list = sorted(glob.glob(pose_path+'/*'))
    poses = []

    resize_factor = (image_size * 1.) / H
    intrinsic = np.loadtxt(os.path.join(intrinsic_path, "intrinsic_color.txt"))
    intrinsic[:2, :] *= resize_factor
    intrinsic[0, 2] -= (W * resize_factor - image_size) * 0.5

    with progress:
        for img_f, pose_f in progress.track(zip(img_file_list, pose_file_list), description=None):
            base_name = os.path.splitext(os.path.basename(img_f))[0]
            save_outputs(img_f, base_name)
            c2w = np.loadtxt(pose_f)
            poses.append(c2w)

    poses = np.array(poses)
    min_vertices = poses[:, :3, 3].min(axis=0)
    max_vertices = poses[:, :3, 3].max(axis=0)
    center = (min_vertices + max_vertices) / 2.
    scale = 2. / (np.max(max_vertices - min_vertices) + 3.)
    # we should normalized to unit cube
    scale_mat = np.eye(4).astype(np.float32)
    scale_mat[:3, 3] = -center
    scale_mat[:3 ] *= scale 
    scale_mat = np.linalg.inv(scale_mat)

    cameras = {}
    for i, img_f in enumerate(img_file_list):
        base_name = os.path.splitext(os.path.basename(img_f))[0]
        cameras[f"scale_mat_{base_name}"] = scale_mat
        cameras[f"world_mat_{base_name}"] = intrinsic @ np.linalg.inv(poses[i])
    
    np.savez(os.path.join(args.output_path, "..", "cameras.npz"), **cameras)


else:
    print("invalid file path!")
    sys.exit()
