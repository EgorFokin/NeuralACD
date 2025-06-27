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

        self.fc1 = VNLinearLeakyReLU(682, 512, dim=3, negative_slope=0.0)
        # self.fc2 = VNLinearLeakyReLU(1024, 512, dim=3, negative_slope=0.0)
        self.fc3 = VNLinearLeakyReLU(512, 256, dim=3, negative_slope=0.0)
        self.fc4 = VNLinearLeakyReLU(256, 64, dim=3, negative_slope=0.0)
        self.fc5 = VNLinearLeakyReLU(64, 16, dim=3, negative_slope=0.0)
        self.fc6 = VNLinear(16, 1)
        
        self.learning_rate = learning_rate
        
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.fc1(x)
        #x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        x = self.fc6(x)
        return x
    
    def compute_loss(self, pred):
        pred = pred.squeeze(1)
        pred_abs = torch.abs(pred)
        
        # pred_abs: [B, 3]
        unit_vectors = torch.eye(3, dtype=pred_abs.dtype, device=pred_abs.device)  # [3, 3]
        # Expand to [B, 3, 3] for broadcasting
        targets = unit_vectors.unsqueeze(0).expand(pred_abs.size(0), -1, -1)  # [B, 3, 3]
        preds = pred_abs.unsqueeze(1).expand(-1, 3, -1)  # [B, 3, 3]

        # Compute per-sample loss to each unit vector
        mse = F.mse_loss(preds, targets, reduction='none').mean(dim=2)  # [B, 3]

        # Pick minimum loss per sample
        loss = mse.min(dim=1).values.mean()  # scalar loss over batch

        return loss
    
    def training_step(self, batch, batch_idx):
        x = batch
        pred = self(x)
        loss = self.compute_loss(pred)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch
        pred = self(x)
        loss = self.compute_loss(pred)
        
        self.log('val_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return [optimizer]