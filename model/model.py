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
    
    def compute_loss(self, pred, target):
        pred = pred.squeeze(1)

        #loss = F.mse_loss(pred, target)
        cosine_sim = F.cosine_similarity(pred, target, dim=-1)
        cosine_sim_neg = F.cosine_similarity(pred, -target, dim=-1)
        max_sim = torch.max(cosine_sim, cosine_sim_neg)
        loss = 1 - max_sim.mean()
        
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.compute_loss(pred, y)
        
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log('lr', self.trainer.optimizers[0].param_groups[0]['lr'], on_step=False,prog_bar=True, on_epoch=True)
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