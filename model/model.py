import torch.nn as nn
import torch.nn.functional as F
import torch
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule

class get_model(nn.Module):
    def __init__(self,num_outputs,normal_channel=True):
        super(get_model, self).__init__()

        self.num_outputs = num_outputs

        self.use_xyz = True

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointnetSAModule(
                npoint=512,
                radius=0.1,
                nsample=64,
                mlp=[0, 32, 32, 64],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=256,
                radius=0.2,
                nsample=64,
                mlp=[64, 64, 64, 128],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                npoint=128,
                radius=0.4,
                nsample=64,
                mlp=[128, 128, 128, 256],
                # bn=False,
                use_xyz=self.use_xyz,
            )
        )

        self.SA_modules.append(
            PointnetSAModule(
                mlp=[256, 256, 512, 1024],
                # bn=False,
                use_xyz=self.use_xyz
            )
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(1024, 512),
            nn.LeakyReLU(True),
            nn.Linear(512, 256),
            nn.LeakyReLU(True),
            nn.Linear(256, self.num_outputs),
            nn.Tanh()
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
        pointcloud = pointcloud.permute(0, 2, 1)
        xyz, features = self._break_up_pc(pointcloud)

        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
            # Normalize predictions and target hyperplanes
            pred_norm = F.normalize(pred, dim=-1)  # Shape: (B, N)
            target_norm = F.normalize(target, dim=-1)  # Shape: (B, M, N)

            # Compute cosine similarity (higher is better, so we minimize 1 - cosine similarity)
            cosine_sim = torch.matmul(pred_norm.unsqueeze(1), target_norm.transpose(-1, -2)).squeeze(1)  # Shape: (B, M)

            # Convert similarity to loss (1 - similarity)
            loss = 1 - cosine_sim  # Shape: (B, M)

            # Take the minimum loss over the M target planes
            min_loss, _ = loss.min(dim=1)  # Shape: (B,)

            return min_loss.mean()  # Return mean loss over the batch

