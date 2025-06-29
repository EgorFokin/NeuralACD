from model.model import PlaneEstimationModel
import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import json
import open3d as o3d
import random
from tqdm import tqdm
import os
from datetime import datetime
import numpy as np
from torch.utils.data import Subset

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

def apply_rotation_to_plane(a,b,c,d,rotation):
    normal = np.array([a, b, c])

    rotation = rotation[:3,:3]
    
    rotated_normal = rotation @ normal

    if np.linalg.norm(normal) == 0:
        raise ValueError("Invalid plane normal (0,0,0).")

    point_on_plane = -d * normal / np.linalg.norm(normal) ** 2 
    rotated_point = rotation @ point_on_plane 

    d_new = -np.dot(rotated_normal, rotated_point)

    if d_new < 0: #make the signs of coeffs consistent
        rotated_normal = -rotated_normal
        d_new = -d_new

    return rotated_normal[0], rotated_normal[1], rotated_normal[2], d_new 

class NeuralACDDataset(Dataset):
    def __init__(self,pc_folder,planes_folder,rotate=True):
        self.rotate =rotate
        with h5py.File(pc_folder, 'r') as f:
            self.data = f['point_clouds'][:]  # shape (N, 512, 3)
            self.hashes = f['hashes'][:]
        with open(planes_folder,'r') as f:
            self.labels = json.load(f)
            
            
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        points = self.data[idx]
        # mesh_hash = self.hashes[idx].decode('utf-8')
        # planes = self.labels[mesh_hash]

        # if self.rotate and random.random() < 0.75:
        #     rotation = o3d.geometry.get_rotation_matrix_from_xyz(np.random.rand(3) * 2 * np.pi)
        # else:
        #     rotation = np.eye(3)

        # points = np.dot(points, rotation[:3,:3].T)

        # rotation = o3d.geometry.get_rotation_matrix_from_xyz(np.random.rand(3) * 2 * np.pi)

        # points = np.dot(points, rotation[:3,:3].T)

        points = points.transpose(1, 0)

        # planes = [apply_rotation_to_plane(*plane[:4],rotation) for plane in planes]
        # planes = np.array(planes)
        label = np.array([0,1,0])
        # label = np.dot(label, rotation[:3,:3].T)


        return points.astype('float32'),label.astype('float32')

train_dataset = NeuralACDDataset("data/train_data.h5","data/plane_cache.json")
val_dataset = NeuralACDDataset("data/val_data.h5","data/plane_cache.json")

train_dataset = Subset(train_dataset, indices=list(range(320)))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=11)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,  num_workers=11)


pl.seed_everything(42)

model = PlaneEstimationModel(learning_rate=1e-3)


callbacks = [
    ModelCheckpoint(monitor='val_loss',
        dirpath='checkpoints/',
        filename='best-model-{epoch:02d}-{val_loss:.2f}',
        save_top_k=3,
        mode='min'),
    LearningRateMonitor()]

logger = CSVLogger("logs", name="my_model")

trainer = pl.Trainer(
        devices="auto",
        accelerator="auto",
        callbacks=callbacks,
        max_epochs=2000,
        log_every_n_steps=100,
        check_val_every_n_epoch=100,
        logger=logger,
    )

# Start Training
trainer.fit(
    model=model,
    train_dataloaders=train_loader,
    val_dataloaders=val_loader
)