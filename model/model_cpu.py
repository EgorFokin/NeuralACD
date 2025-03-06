import torch.nn as nn
import torch.nn.functional as F
import torch
from pointnet2_utils import PointNetSetAbstraction

class get_model(nn.Module):
    def __init__(self,num_outputs,normal_channel=True):
        super(get_model, self).__init__()

        self.num_outputs = num_outputs

        self.use_xyz = True

        self._build_model()

    def _build_model(self):
        self.SA_modules = nn.ModuleList()
        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=512,
                radius=0.1,
                nsample=64,
                in_channel=3,
                mlp=[32, 32, 64],
                group_all=False,
            )
        )

        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=256,
                radius=0.2,
                nsample=64,
                in_channel=64+3,
                mlp=[ 64, 64, 128],
                group_all=False,
            )
        )

        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=128,
                radius=0.4,
                nsample=64,
                in_channel=128+3,
                mlp=[ 128, 128, 256],
                group_all=False,
            )
        )

        self.SA_modules.append(
            PointNetSetAbstraction(
                npoint=None,
                radius=None,
                nsample=None,
                in_channel=256+3,
                mlp=[ 256, 512, 1024],
                group_all=True,
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
        # pointcloud = pointcloud.permute(0, 2, 1)
        # xyz, features = self._break_up_pc(pointcloud)
        xyz = pointcloud
        features = None
        for module in self.SA_modules:
            xyz, features = module(xyz, features)

        return self.fc_layer(features.squeeze(-1))


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
            # # Normalize predictions and target hyperplanes
            # pred_norm = F.normalize(pred, dim=-1)  # Shape: (B, N)
            # target_norm = F.normalize(target, dim=-1)  # Shape: (B, M, N)

            # # Compute cosine similarity (higher is better, so we minimize 1 - cosine similarity)
            # cosine_sim = torch.matmul(pred_norm.unsqueeze(1), target_norm.transpose(-1, -2)).squeeze(1)  # Shape: (B, M)

            # # Convert similarity to loss (1 - similarity)
            # loss = 1 - cosine_sim  # Shape: (B, M)

            # # Take the minimum loss over the M target planes
            # min_loss, _ = loss.min(dim=1)  # Shape: (B,)

            # return min_loss.mean()  # Return mean loss over the batch



            target_norm = F.normalize(target, dim=-1)
            #target_norm[..., 3] = target_norm[..., 3] * 2 - 1 # d is always positive, so we need to convert it to the range [-1, 1]


            #distance to the closest plane
            loss_fn = nn.MSELoss(reduction='none')  # Compute element-wise MSE

            # Compute MSE loss for all target planes
            loss = loss_fn(pred.unsqueeze(1), target_norm)  # Shape: (B, M, 4)

            # Sum over the last dimension (MSE is applied to 4D vectors)
            loss = loss.mean(dim=-1)  # Shape: (B, M)

            # Take the minimum loss over the M target planes for each batch
            min_loss, _ = loss.min(dim=1)  # Shape: (B,)

            return min_loss.mean()  # Return mean loss over the batch
