import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.vn_layers import *
from model.utils.vn_pointnet import PointNetEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PlaneEstimationModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        
        # Initialize model components
        args = type('', (), {})()
        args.n_knn = 20
        args.pooling = 'max'
        args.num_point = 512
        args.rot = "aligned"
        
        channel = 3
        self.feat = PointNetEncoder(args, global_feat=True, feature_transform=True, channel=channel)

        self.fc1 = VNLinearLeakyReLU(682, 1024, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(1024, 512, dim=3, negative_slope=0.0)
        self.fc3 = VNLinearLeakyReLU(512, 256, dim=3, negative_slope=0.0)
        self.fc4 = VNLinearLeakyReLU(256, 64, dim=3, negative_slope=0.0)
        self.fc5 = VNLinearLeakyReLU(64, 16, dim=3, negative_slope=0.0)
        self.fc6 = VNLinear(16, 2)
        
        self.learning_rate = learning_rate
        
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    
    def compute_loss(self, pred, target):
        # Distance term
        pred_d = -torch.sum(pred[:, 1, :] * pred[:, 0, :], dim=1)
        diff = target[..., 3] - pred_d.unsqueeze(1)
        squared_loss = diff**2
        
        # Normal direction term
        pred_dir = pred[:, 0, :]
        target_dir = target[..., :3]
        pred_dir_norm = F.normalize(pred_dir, dim=-1)
        target_dir_norm = F.normalize(target_dir, dim=-1)
        cosine_sim = torch.matmul(pred_dir_norm.unsqueeze(1), target_dir_norm.transpose(-1, -2)).squeeze(1)
        cosine = 1 - cosine_sim
        
        # Combined loss
        loss = squared_loss + cosine
        min_loss, _ = loss.min(dim=1)
        normalization_loss = torch.abs((1 - torch.norm(pred_dir, dim=-1)))
        
        return min_loss.mean() + normalization_loss.mean()
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.compute_loss(pred, y)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.compute_loss(pred, y)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]