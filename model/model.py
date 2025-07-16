import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.vn_layers import *
from model.utils.vn_pointnet import PointNetEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModuleMSG
from torch.optim.lr_scheduler import LambdaLR

class ACDModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, weight_decay=1e-2, use_xyz=True):
        super().__init__()
        self.save_hyperparameters()
        self._build_model()

        LOSS_ALPHA = 2
        LOSS_BETA = 1

        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([LOSS_ALPHA/LOSS_BETA]))
        
        
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        c_in = 0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=4096,
                radii=[0.02, 0.03, 0.05],
                nsamples=[8 ,16, 32],
                mlps=[[c_in, 8, 8, 16], [c_in, 16, 16, 32], [c_in, 32, 32, 64]],
                use_xyz=self.hparams.use_xyz,
            )
        )
        c_out_0 = 16 + 32 + 64

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=1024,
                radii=[0.05, 0.1],
                nsamples=[16, 32],
                mlps=[[c_in, 64, 64, 128], [c_in, 64, 96, 128]],
                use_xyz=self.hparams.use_xyz,
            )
        )
        c_out_1 = 128 + 128

        c_in = c_out_1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.25],
                nsamples=[16, 32],
                mlps=[[c_in, 128, 196, 256], [c_in, 128, 196, 256]],
                use_xyz=self.hparams.use_xyz,
            )
        )
        c_out_2 = 256 + 256

        # c_in = c_out_2
        # self.SA_modules.append(
        #     PointnetSAModuleMSG(
        #         npoint=16,
        #         radii=[0.4, 0.8],
        #         nsamples=[16, 32],
        #         mlps=[[c_in, 256, 256, 512], [c_in, 256, 384, 512]],
        #         use_xyz=self.hparams.use_xyz,
        #     )
        # )
        # c_out_3 = 512 + 512
        



        self.FP_modules = nn.ModuleList()
        self.FP_modules.append(PointnetFPModule(mlp=[256, 128, 128]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_0, 256, 256]))
        self.FP_modules.append(PointnetFPModule(mlp=[512 + c_out_1, 512, 512]))
        # self.FP_modules.append(PointnetFPModule(mlp=[c_out_3 + c_out_2, 512, 512]))

        self.fc_layer = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Conv1d(128, 1, kernel_size=1),
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features

        
    
    def forward(self, pointcloud):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)

        l_xyz, l_features = [xyz], [features]
        for i in range(len(self.SA_modules)):
            li_xyz, li_features = self.SA_modules[i](l_xyz[i], l_features[i])
            l_xyz.append(li_xyz)
            l_features.append(li_features)

        for i in range(-1, -(len(self.FP_modules) + 1), -1):
            l_features[i - 1] = self.FP_modules[i](
                l_xyz[i - 1], l_xyz[i], l_features[i - 1], l_features[i]
            )

        return self.fc_layer(l_features[0])

    
    def compute_loss(self, pred, target):
        pred = pred.squeeze(1)

        #loss = F.mse_loss(pred, target)
        loss = self.loss(pred, target)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.compute_loss(pred, y)

        self.log('train_loss', loss, prog_bar=True)
        
        alpha = 0.05
        self.previous_ema_loss = getattr(self, 'previous_ema_loss', loss)
        self.ema_loss = ema_loss = alpha * loss + (1 - alpha) * self.previous_ema_loss
        self.previous_ema_loss = ema_loss
        self.log('ema_loss', ema_loss, prog_bar=True)
        

        return loss
    
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        def lr_lambda(current_step):
            # Decrease LR to 10% over 500 steps using exponential decay
            decay_rate = 0.1 ** (1 / 500)
            return decay_rate ** current_step

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency":1
            }
        }
