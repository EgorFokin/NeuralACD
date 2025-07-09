from model.model import ACDModel
import h5py
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn as nn
import json
import open3d as o3d
import random
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np
from torch.utils.data import Subset
import lib_acd_gen

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


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
            
    def __iter__(self):
        num_spheres = random.randint(1, 20)
        while True:

            structure_type = random.choice(['sphere', 'cuboid'])
            if structure_type == 'sphere':
                structure = lib_acd_gen.generate_sphere_structure(num_spheres)
            elif structure_type == 'cuboid':
                structure = lib_acd_gen.generate_cuboid_structure(num_spheres)


            verts = []
            triangles = []
            cut_verts = []
            vertex_offset = 0
            for mesh in structure:
                verts.extend(mesh.vertices)
                triangles.extend([[v0 + vertex_offset, v1 + vertex_offset, v2 + vertex_offset] for v0, v1, v2 in mesh.triangles])

                vertex_offset += len(mesh.vertices)
                cut_verts.extend(mesh.cut_verts)
            
            o3d_mesh = o3d.geometry.TriangleMesh()
            
            #print(verts)
            o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)

            pcd = o3d_mesh.sample_points_uniformly(number_of_points=10000)
            points = np.asarray(pcd.points)
            distances = self.get_distances(points, cut_verts)

            points = torch.tensor(points, dtype=torch.float32)
            distances = torch.tensor(distances, dtype=torch.float32)
            distances = torch.clamp(distances, 0.0, 0.05)*20.0  # Scale distances to [0, 1] range

            yield points, distances

dataset = ACDgen()

#train_dataset = Subset(train_dataset, indices=list(range(320)))

train_loader = DataLoader(dataset, batch_size=16, num_workers=11)

sample = next(iter(train_loader))
print("Sample points shape:", sample[0].shape)
print("Sample distances shape:", sample[1].shape)

pl.seed_everything(42)
torch.set_float32_matmul_precision('high')

model = ACDModel(learning_rate=1e-4)



callbacks = [
    ModelCheckpoint(monitor='train_loss',
        dirpath=f'checkpoints/{str(datetime.now().strftime("%d,%m,%Y-%H:%M:%S"))}/',
        filename='best-model-{train_loss}',
        save_top_k=3,
        mode='min',
        every_n_train_steps=100),
        ]

logger = CSVLogger("logs", name="my_model")

trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        callbacks=callbacks,
        log_every_n_steps=10,
        logger=logger,
    )

# Start Training
trainer.fit(
    model=model,
    train_dataloaders=train_loader
)