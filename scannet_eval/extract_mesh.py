import sys

import argparse
from pyhocon import ConfigFactory
import trimesh
import numpy as np
import open3d as o3d
import torch

import os
import glob

from evaluate import refuse
from tsdf import TSDF

# hard-coded image size
H, W = 480, 640


def load_poses(pose_path):
    pose_files = sorted(glob.glob(os.path.join(pose_path, '*.txt')))
    poses = []
    for f in pose_files:
        c2w = np.loadtxt(f)
        poses.append(c2w)
    poses = np.array(poses)
    return poses


def arg_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/dtu.conf')
    parser.add_argument('--scan_id', type=str, default="scene0488_01", help='If set, taken to be the scan id.')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')

    return parser.parse_args()

def main(opt):
    conf = ConfigFactory.parse_file(opt.conf)
    root_dir = os.path.join(conf['dataset']['omnidata_dir'], opt.scan_id, 'MonoSDF')
    ply_file = os.path.join(root_dir, 'plots', f'{opt.nepoch}', f'surface_{opt.nepoch}.ply')
    out_path = os.path.join(root_dir, 'seq_ransac')
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    mesh = trimesh.load(ply_file)

    # transform to world coordinate
    cam_file = np.load(os.path.join(root_dir, 'omnidata', 'cameras.npz'))
    scale_mat = cam_file[cam_file.files[0]]
    mesh.vertices = (scale_mat[:3, :3] @ mesh.vertices.T + scale_mat[:3, 3:]).T

    mesh.export(os.path.join(out_path, f'MonoSDF-{opt.scan_id}-raw.ply'))

    # load pose and intrinsic for render depth
    poses = load_poses(os.path.join(conf['dataset']['data_dir'], opt.scan_id, 'pose'))
    intrinsic_path = os.path.join(conf['dataset']['data_dir'], opt.scan_id, 'intrinsic', 'intrinsic_color.txt')
    K = np.loadtxt(intrinsic_path)[:3, :3]

    mesh = refuse(mesh, poses, K, H, W, voxel_length=0.02)

    out_mesh_path = os.path.join(out_path, f'MonoSDF-{opt.scan_id}.ply')
    o3d.io.write_triangle_mesh(out_mesh_path, mesh)
    mesh = trimesh.load(out_mesh_path)

    tsdf_pred = TSDF.from_mesh(mesh, voxel_size=0.01)

    device = tsdf_pred.origin.device
    verts = torch.from_numpy(np.asarray(mesh.vertices)).to(device)
    coords = torch.round((verts - tsdf_pred.origin) / tsdf_pred.voxel_size).clamp(
                min = torch.zeros((1, 3), device=device), max = torch.tensor(tsdf_pred.tsdf_values.shape, device=device).view(1, 3) - 1
            ).long()
    tsdf_pred.tsdf_values[coords[:, 0], coords[:, 1], coords[:, 2]] = 0.
    data = {'origin': tsdf_pred.origin.detach().numpy(),
            'voxel_size': np.array([tsdf_pred.voxel_size] * 3),
            'tsdf': tsdf_pred.tsdf_values.detach().numpy()}
    np.savez_compressed(os.path.join(out_path, f'MonoSDF-{opt.scan_id}.npz'), **data)


if __name__ == '__main__':
    opt = arg_parsing()
    main(opt)