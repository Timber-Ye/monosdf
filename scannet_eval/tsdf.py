import torch
import trimesh
import numpy as np
from typing import Tuple

import os
from skimage import measure

class TSDF:

    """
        Class for housing and data handling TSDF volumes.
    """
    # Ensures the final voxel volume dimensions are multiples of 8
    VOX_MOD = 8

    def __init__(
            self,
            voxel_coords: torch.tensor,
            tsdf_values: torch.tensor,
            tsdf_weights: torch.tensor,
            voxel_size: float,
            origin: torch.tensor,
        ):
        """
            Sets interal class attributes.
        """
        self.voxel_coords = voxel_coords.half()
        self.tsdf_values = tsdf_values.half()
        self.tsdf_weights = tsdf_weights.half()
        self.voxel_size = voxel_size
        self.origin = origin.half()

    @classmethod
    def from_file(cls, tsdf_file):
        """ Loads a tsdf from a numpy file. """
        tsdf_data = np.load(tsdf_file)

        tsdf_values = torch.from_numpy(tsdf_data['tsdf_values'])
        origin = torch.from_numpy(tsdf_data['origin'])
        voxel_size = tsdf_data['voxel_size'].item()

        tsdf_weights = torch.ones_like(tsdf_values)

        voxel_coords = cls.generate_voxel_coords(origin, tsdf_values.shape[1:], voxel_size)

        return TSDF(voxel_coords, tsdf_values, tsdf_weights, voxel_size)

    @classmethod
    def from_mesh(cls, mesh: trimesh.Trimesh, voxel_size: float):
        """ Gets TSDF bounds from a mesh file. """
        xmax, ymax, zmax = mesh.vertices.max(0)
        xmin, ymin, zmin = mesh.vertices.min(0)

        bounds = {'xmin': xmin, 'xmax': xmax,
                  'ymin': ymin, 'ymax': ymax,
                  'zmin': zmin, 'zmax': zmax}

        # create a buffer around bounds
        for key, val in bounds.items():
            if 'min' in key:
                bounds[key] = val - 3 * voxel_size
            else:
                bounds[key] = val + 3 * voxel_size
        return cls.from_bounds(bounds, voxel_size)

    @classmethod
    def from_bounds(cls, bounds: dict, voxel_size: float):
        """ Creates a TSDF volume with bounds at a specific voxel size. """

        expected_keys = ['xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax']
        for key in expected_keys:
            if key not in bounds.keys():
                raise KeyError("Provided bounds dict need to have keys"
                               "'xmin', 'xmax', 'ymin', 'ymax', 'zmin', 'zmax'!")

        num_voxels_x = int(
            np.ceil((bounds['xmax'] - bounds['xmin']) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD
        num_voxels_y = int(
            np.ceil((bounds['ymax'] - bounds['ymin']) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD
        num_voxels_z = int(
            np.ceil((bounds['zmax'] - bounds['zmin']) / voxel_size / cls.VOX_MOD)) * cls.VOX_MOD

        origin = torch.FloatTensor([bounds['xmin'], bounds['ymin'], bounds['zmin']])

        voxel_coords = cls.generate_voxel_coords(
            origin, (num_voxels_x, num_voxels_y, num_voxels_z), voxel_size).half()

        # init to -1s
        tsdf_values = -torch.ones_like(voxel_coords[0]).half()
        tsdf_weights = torch.zeros_like(voxel_coords[0]).half()

        return TSDF(voxel_coords, tsdf_values, tsdf_weights, voxel_size, origin)

    @classmethod
    def generate_voxel_coords(cls,
                              origin: torch.tensor,
                              volume_dims: Tuple[int, int, int],
                              voxel_size: float):
        """ Gets world coordinates for each location in the TSDF. """

        grid = torch.meshgrid([torch.arange(vd) for vd in volume_dims])

        voxel_coords = origin.view(3, 1, 1, 1) + torch.stack(grid, 0) * voxel_size

        return voxel_coords


    def cuda(self):
        """ Moves TSDF to gpu memory. """
        self.voxel_coords = self.voxel_coords.cuda()
        self.tsdf_values = self.tsdf_values.cuda()
        if self.tsdf_weights is not None:
            self.tsdf_weights = self.tsdf_weights.cuda()

    def cpu(self):
        """ Moves TSDF to cpu memory. """
        self.voxel_coords = self.voxel_coords.cpu()
        self.tsdf_values = self.tsdf_values.cpu()
        if self.tsdf_weights is not None:
            self.tsdf_weights = self.tsdf_weights.cpu()

    def to_mesh(self, scale_to_world=True, export_single_mesh=False):
        """ Extracts a mesh from the TSDF volume using marching cubes. 
        
            Args:
                scale_to_world: should we scale vertices from TSDF voxel coords 
                    to world coordinates?
                export_single_mesh: returns a single walled mesh from marching
                    cubes. Requires a custom implementation of 
                    measure.marching_cubes that supports single_mesh
                
        """
        tsdf = self.tsdf_values.detach().cpu().clone().float()
        tsdf_np = tsdf.clamp(-1, 1).cpu().numpy()

        if export_single_mesh:
            verts, faces, norms, _ = measure.marching_cubes(
                                                    tsdf_np, 
                                                    level=0, 
                                                    allow_degenerate=False, 
                                                    single_mesh = True,
                                                )
        else:
            verts, faces, norms, _ = measure.marching_cubes(
                                                    tsdf_np,
                                                    level=0, 
                                                    allow_degenerate=False,
                                                )

        if scale_to_world:
            verts = self.origin.cpu().view(1, 3) + verts * self.voxel_size

        mesh = trimesh.Trimesh(vertices=verts, faces=faces, normals=norms)
        return mesh

    def save(self, savepath, filename, save_mesh=True):
        """ Saves a mesh to disk. """
        self.cpu()
        os.makedirs(savepath, exist_ok=True)
        
        if save_mesh:
            mesh = self.to_mesh()
            trimesh.exchange.export.export_mesh(
                mesh, os.path.join(savepath, 
                                    filename).replace(".bin", ".ply"), "ply")
        else:
            data = {'origin': self.origin.numpy(),
                    'voxel_size': self.voxel_size,
                    'tsdf': self.tsdf_values.detach().numpy()}
            np.savez_compressed(os.path.join(savepath, filename), **data)