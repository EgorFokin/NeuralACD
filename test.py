import lib_acd_gen
import open3d as o3d
import numpy as np
from matplotlib import cm
import trimesh
import torch
import sys

from train import ACDgen

from model.model import ACDModel

CHECKPOINT="checkpoints/13,07,2025-15:18:16/best-model-ema_loss=0.3352755010128021.ckpt"

def normalize_points(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_distance = np.max(np.linalg.norm(points, axis=1))
    points /= max_distance
    pcd.points = o3d.utility.Vector3dVector(points)



it = ACDgen().__iter__()
lib_acd_gen.set_seed(3)
#next(it)  
points, distances_t = next(it)

# with open("points.npy", "rb") as f:
#     points = np.load(f)
# with open("distances.npy", "rb") as f:
#     distances_t = np.load(f)

# o3d_mesh = o3d.io.read_triangle_mesh("data/meshes/cow.obj")

# pcd = o3d_mesh.sample_points_uniformly(number_of_points=10000)
# normalize_points(pcd)

#points =np.asarray(pcd.points) 

with torch.no_grad():
    model = ACDModel().cuda()
    model.load_state_dict(torch.load(CHECKPOINT)["state_dict"])
    model.eval()
    distances = model(torch.tensor(points, dtype=torch.float32).cuda().unsqueeze(0))
    
    

    loss = model.compute_loss(distances, torch.tensor(distances_t, dtype=torch.float32).cuda().unsqueeze(0))
    print("Loss:", loss.item())

    distances = torch.sigmoid(distances)  # Ensure distances are in [0, 1] range
    distances = distances.squeeze().cpu().numpy()


#distances = get_distances(pcd.points, cut_verts)
distances = distances


# threshold = 0.5
# distances[distances < threshold] = 0
# distances[distances >= threshold] = 0.8


# Map to color (e.g., blue → red gradient using matplotlib)
colormap = cm.get_cmap("jet")
colors = colormap(distances)[:, :3]  # RGB, invert to make close = red

alpha = np.clip(distances, 0.5, 1)
rgba = np.hstack([colors, alpha[:, np.newaxis]])

point_cloud = trimesh.points.PointCloud(points, colors=rgba)

point_cloud.show()