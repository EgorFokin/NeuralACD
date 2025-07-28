import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.vn_layers import *
from model.utils.vn_pointnet import PointNetEncoder
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pointnet2_ops.pointnet2_modules import PointnetFPModule#, PointnetSAModuleMSG
from torch.optim.lr_scheduler import LambdaLR


from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops import pointnet2_utils  

def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)

import pointnet2_ops._ext as _ext
def debug_radius_coverage(new_xyz, xyz, radius, nsample):
    B, N, _ = xyz.shape
    neighbors = _ext.ball_query(new_xyz, xyz, radius, nsample)
    print(f"Radius {radius}, NSample {nsample}:")
    for b in range(B):
        for i in range(new_xyz.shape[1]):
            num_neighbors = (neighbors[b, i] >= 0).sum().item()
            for neighbor in neighbors[b, i]:
                if neighbor >= 0:
                    dist2 = torch.sqrt(torch.sum((xyz[b, neighbor] - new_xyz[b, i]) ** 2))
                    print(f"  Point {i} in batch {b} has {num_neighbors} neighbors, distance: {dist2.item()}")

class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)

            # if self.groupers[i].radius == 0.1:
            #     print(features.shape)
            #     print(new_features.shape)
            #     for i in range(5):
            #         print(new_features[0, :3, i, :])
            #     exit()

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)



class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3

            self.mlps.append(build_shared_mlp(mlp_spec, bn))


class ACDModel(pl.LightningModule):
    def __init__(self, learning_rate=1e-3, weight_decay=1e-2, use_xyz=True):
        super().__init__()
        self.save_hyperparameters()
        self._build_model()

        LOSS_ALPHA = 10
        LOSS_BETA = 1

        self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([LOSS_ALPHA/LOSS_BETA]))
        
        
    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        c_in = 1
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=10000,
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
                npoint=2500,
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
        self.FP_modules.append(PointnetFPModule(mlp=[256 + 1, 128, 128]))
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

        
    
    def forward(self, pointcloud, apply_sigmoid=False):
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

        fc_out = self.fc_layer(l_features[0])
        fc_out = fc_out.squeeze(1)  # (B, N)

        if apply_sigmoid:
            fc_out = torch.sigmoid(fc_out)

        return fc_out

    
    def compute_loss(self, pred, target):
        #loss = F.mse_loss(pred, target)
        loss = self.loss(pred, target)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        pred = self(x)
        loss = self.compute_loss(pred, y)

        self.log('train_loss', loss, prog_bar=True)

        # print(print(y.mean()))
        # print(print(torch.sigmoid(pred).mean()))
        
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
