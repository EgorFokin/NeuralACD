from model.model import ACDModel
import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader, TensorDataset
import torch.nn as nn
import json
import open3d as o3d
import random
from tqdm import tqdm
import os
from datetime import datetime
import shutil
import numpy as np
from torch.utils.data import Subset
import lib_acd_gen

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.profilers import AdvancedProfiler


class ACDgen(IterableDataset):
    def __init__(self):
        pass

    def get_distances(self, pcd_points,cut_points):
        if len(cut_points) == 0:
            return np.zeros(len(pcd_points), dtype=np.float32)
        # Create a KD-tree from x
        cut_pcd = o3d.geometry.PointCloud()
        cut_pcd.points = o3d.utility.Vector3dVector(cut_points)
        kdtree = o3d.geometry.KDTreeFlann(cut_pcd)

        distances = []

        for p in pcd_points:
            _, idx, dist2 = kdtree.search_knn_vector_3d(p, 1)
            distances.append(np.sqrt(dist2[0]))

        distances = np.array(distances)
        
        return distances
    
    def apply_gaussian_filter(self, pcd, sigma=0.02, radius = 0.05):
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        smoothed_points = np.zeros_like(points)

        for i in range(len(points)):
            [_, idxs, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius)
            neighbors = points[idxs]
            distances = np.linalg.norm(neighbors - points[i], axis=1)
            weights = np.exp(-distances**2 / (2 * sigma**2))
            weights /= np.sum(weights)
            smoothed_points[i] = np.sum(neighbors * weights[:, np.newaxis], axis=0)

        pcd.points = o3d.utility.Vector3dVector(smoothed_points)

    def apply_random_transform(self, pcd):
        scale_x = np.random.uniform(0.5, 4.0)
        scale_y = np.random.uniform(0.5, 4.0)
        scale_z = np.random.uniform(0.5, 4.0)

        stretch_matrix = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, scale_z]
        ])

        points = np.asarray(pcd.points)
        points = points @ stretch_matrix.T

        pcd.points = o3d.utility.Vector3dVector(points)
    
    def apply_random_rotation(self, pcd):
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        points = np.asarray(pcd.points)
        points = points @ rotation_matrix.T

        pcd.points = o3d.utility.Vector3dVector(points)

    def normalize_points(self, pcd):
        points = np.asarray(pcd.points)
        centroid = np.mean(points, axis=0)
        points -= centroid
        max_distance = np.max(np.linalg.norm(points, axis=1))
        points /= max_distance
        pcd.points = o3d.utility.Vector3dVector(points)
            
    def __iter__(self):
        num_spheres = 5#random.randint(3, 20)
        while True:
            structure_type = "sphere" #random.choice(['sphere', 'cuboid'])
            if structure_type == 'sphere':
                structure = lib_acd_gen.generate_sphere_structure(num_spheres)
            elif structure_type == 'cuboid':
                structure = lib_acd_gen.generate_cuboid_structure(num_spheres)


            o3d_mesh = o3d.geometry.TriangleMesh()
            
            #print(verts)
            o3d_mesh.vertices = o3d.utility.Vector3dVector(structure.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(structure.triangles)

            pcd = o3d_mesh.sample_points_uniformly(number_of_points=10000)
            
            distances = self.get_distances(np.asarray(pcd.points), structure.cut_verts)

            # self.apply_random_transform(pcd)
            # self.apply_random_rotation(pcd)
            # self.apply_gaussian_filter(pcd,sigma=0.1,radius=random.uniform(0.01, 0.15))
            self.normalize_points(pcd)
            points = np.asarray(pcd.points)

            points = torch.tensor(points, dtype=torch.float32)
            distances = torch.tensor(distances, dtype=torch.float32)
            distances = 1 - torch.clamp(distances, 0.0, 0.05)*20.0  # Scale distances to [0, 1] range

            yield points, distances

if __name__ == "__main__":
    dataset = ACDgen()



    train_loader = DataLoader(dataset, batch_size=8, num_workers=8)


    pl.seed_everything(42)
    
    torch.set_float32_matmul_precision('high')

    model = ACDModel(learning_rate=1e-3)

    #copy the model.py into checkpoint directory
    os.makedirs(f'checkpoints/{str(datetime.now().strftime("%d,%m,%Y-%H:%M:%S"))}/', exist_ok=True)
    shutil.copy('model/model.py', f'checkpoints/{str(datetime.now().strftime("%d,%m,%Y-%H:%M:%S"))}/model.py')

    profiler = AdvancedProfiler(dirpath="profiler_logs", filename=str(datetime.now().strftime("%d,%m,%Y-%H:%M:%S")))

    callbacks = [
        ModelCheckpoint(monitor='ema_loss',
            dirpath=f'checkpoints/{str(datetime.now().strftime("%d,%m,%Y-%H:%M:%S"))}/',
            filename='best-model-{ema_loss}',
            save_top_k=3,
            mode='min',
            every_n_train_steps=1),
            ]

    logger = CSVLogger("logs")

    trainer = pl.Trainer(
            devices=1,
            accelerator="auto",
            callbacks=callbacks,
            log_every_n_steps=10,
            logger=logger,
            max_steps=2000,
            #profiler=profiler,
        )

    # Start Training
    trainer.fit(
        model=model,
        train_dataloaders=train_loader
    )
