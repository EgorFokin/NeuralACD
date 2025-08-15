import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..","lib", "build"))

import lib_neural_acd
import torch
from torch.utils.data import Dataset
import open3d as o3d
import trimesh
import numpy as np

from utils.misc import get_lib_mesh, normalize_mesh


# BLACKLIST = ["face-YH.off", "mask.off"]


class VHACD(Dataset):
    def __init__(self, config, path = 'data/v-hacd-data/data'):
        self.config = config
        self.paths = []
        for file in os.listdir(path):
            if file.endswith('.off'):
                self.paths.append(os.path.join(path,file))

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        mesh = trimesh.load(path, force='mesh')
        mesh = get_lib_mesh(mesh)
        normalize_mesh(mesh)
        lib_neural_acd.preprocess(mesh, 50.0, 0.55)
        
        points = lib_neural_acd.VecArray3d()
        point_tris = lib_neural_acd.VecInt()
        mesh.extract_point_set(points, point_tris, self.config.general.num_points)

        points = np.asarray(points)

        tmesh = trimesh.Trimesh(vertices=np.asarray(mesh.vertices), faces=np.asarray(mesh.triangles))
        curvature = trimesh.curvature.discrete_gaussian_curvature_measure(tmesh, points, radius=0.02)
        normals = tmesh.face_normals[np.asarray(point_tris)]
        points = np.hstack((points, curvature[:, np.newaxis],normals))

        points = torch.tensor(points, dtype=torch.float32)




        return points, mesh, os.path.splitext(os.path.basename(path))[0]



