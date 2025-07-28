import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..","lib", "build"))

import lib_neural_acd
import torch
from torch.utils.data import Dataset
import open3d as o3d
import trimesh
import numpy as np

from utils.misc import get_lib_mesh, normalize_mesh, get_point_cloud


class VHACD(Dataset):
    def __init__(self, path = 'data/v-hacd-data/data'):
        self.paths = []
        for file in os.listdir(path):
            if file.endswith('.off'):
                self.paths.append(os.path.join(path,file))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        print(path)
        mesh = trimesh.load(path, force='mesh')
        mesh = get_lib_mesh(mesh)
        mesh = normalize_mesh(mesh)
        lib_neural_acd.preprocess(mesh, 50.0, 0.05)
        pcd = get_point_cloud(mesh)
        points = pcd.points

        tmesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
        curvature = trimesh.curvature.discrete_gaussian_curvature_measure(tmesh, points, radius=0.02)
        points = np.hstack((points, curvature[:, np.newaxis]))

        points = torch.tensor(points, dtype=torch.float32)




        return points, mesh



