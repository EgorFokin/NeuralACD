import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from vn_layers import *
from vn_pointnet import PointNetEncoder

class get_model(nn.Module):
    def __init__(self, normal_channel=True):
        args =  type('', (), {})()  # Create a simple object to hold attributes

        args.n_knn = 20
        args.pooling = 'max'
        args.num_point = 512
        args.rot = "aligned"

        super(get_model, self).__init__()
        self.args = args
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = PointNetEncoder(args, global_feat=True, feature_transform=True, channel=channel)
        self.fc1 = VNLinearLeakyReLU(682,256, dim=3, negative_slope=0.0)
        self.fc2 = VNLinearLeakyReLU(256,128, dim=3,negative_slope=0.0)
        self.fc3 = VNLinearLeakyReLU(128,64, dim=3,negative_slope=0.0)
        self.fc4 = VNLinearLeakyReLU(64, 32, dim=3,negative_slope=0.0)
        self.fc5 = VNLinear(32, 2)
    def forward(self, x):
        x, trans, trans_feat = self.feat(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        x = self.fc5(x)
        return x
    


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
            #convert pred to ax+by+cz+d format
            #pred is of shape (B, 2,3)


            #target_norm[..., 3] = target_norm[..., 3] * 2 - 1 # d is always positive, so we need to convert it to the range [-1, 1]


            # #distance to the closest plane
            # loss_fn = nn.MSELoss(reduction='none')  # Compute element-wise MSE

            # # Compute MSE loss for all target planes
            # loss = loss_fn(pred.unsqueeze(1), target)  # Shape: (B, M, 4)

            # # Sum over the last dimension (MSE is applied to 4D vectors)
            # mse = loss.mean(dim=-1)  # Shape: (B, M)

            pred_d = -torch.sum(pred[:, 1, :]* pred[:, 0, :],dim=1)  # Compute the dot product of the two vectors
            diff = target[..., 3] - pred_d.unsqueeze(1)  # Shape:  (B, M, D)
            squared_loss = diff**2



            #compare normal directions
            pred_dir = pred[:,0,:]
            target_dir = target[..., :3]  # Shape: (B, M, 3)

            # Normalize the direction vectors to compare only their directions
            pred_dir_norm = F.normalize(pred_dir, dim=-1)  # Shape: (B, 3)
            target_dir_norm = F.normalize(target_dir, dim=-1)  # Shape: (B, M, 3)

            # Compute cosine similarity (higher means better alignment)
            cosine_sim = torch.matmul(pred_dir_norm.unsqueeze(1), target_dir_norm.transpose(-1, -2)).squeeze(1)  # Shape: (B, M)

            # Convert similarity to loss (1 - similarity, so lower is better)
            cosine = 1 - cosine_sim  # Shape: (B, M)

            

            loss = squared_loss + cosine

            # Take the minimum loss over the M target planes
            min_loss, _ = loss.min(dim=1)  # Shape: (B,)
            normalization_loss = torch.abs((1 - torch.norm(pred_dir, dim=-1)))


            return min_loss.mean() + normalization_loss.mean()
