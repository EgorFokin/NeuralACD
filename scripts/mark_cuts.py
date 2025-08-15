import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..","lib", "build"))
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import trimesh
import torch
from utils.ACDgen import ACDgen
from model.model import ACDModel
import lib_neural_acd
import argparse
from utils.misc import *
from utils.visualization import save_rotating_pcd_gif, show_pcd

def normalize_points(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_distance = np.max(np.linalg.norm(points, axis=1))
    points /= max_distance
    pcd.points = o3d.utility.Vector3dVector(points)

def load_model(checkpoint):
    model = ACDModel().cuda()
    state_dict = torch.load(checkpoint, weights_only=True)["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model.cuda()

def threshold_values(values, config):
    for i in range(len(values)):
        values[i][values[i] < config.general.cut_point_threshold] = 0

        if len(np.flatnonzero(values[i])) > config.general.cut_point_limit:
            # Keep only the top cut points based on distances
            # selected = np.argpartition(distances[i], -config.general.cut_point_limit)[-config.general.cut_point_limit:]

            nonzero_indices = np.flatnonzero(values[i])

            selected = np.random.choice(nonzero_indices, size=config.general.cut_point_limit, replace=False)


            values[i][selected] = 1
            values[i][values[i] < 1] = 0
        else:
            # If fewer than cut_point_limit points, set all to 1
            values[i][values[i] > 0] = 1

    return values

def mark_cuts(points, checkpoint, config, no_threshold=False):

    if isinstance(points, np.ndarray):
        points = torch.tensor(points, dtype=torch.float32)
    
    if isinstance(points, list):
        points = torch.stack([torch.tensor(p, dtype=torch.float32) for p in points], dim=0)

    batched = True
    points = points.cuda()

    if points.ndim == 2:
        points = points.unsqueeze(0)
        batched = False

    model = load_model(checkpoint)

    with torch.no_grad():

        values = []

        for start in range(0, points.shape[0], config.model.batch_size):
            end = min(start + config.model.batch_size, points.shape[0])
            batch_points = points[start:end]
    
            pred = model(batch_points, apply_sigmoid=True)

            values.append(pred)
        values = torch.cat(values, dim=0)

        

        values = values.cpu().numpy()

    if no_threshold:
        if not batched:
            values = values.squeeze(0)
        return values

    values = threshold_values(values, config)

    if not batched:
        values = values.squeeze(0)

    return values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark cuts in point cloud using ACD model.")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoint.ckpt", help="Path to the model checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--gt", action="store_true", help="Use ground truth points instead of model predictions.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file.")
    parser.add_argument("--no-threshold", action="store_true", help="Do not apply thresholding to cut points.")
    parser.add_argument("--path", type=str, default="", help="Path to the mesh file.")
    
    args = parser.parse_args()
    
    config = load_config(args.config)

    if args.path:
        mesh = trimesh.load(args.path, force='mesh')
        structure, points = load_mesh(mesh,config)

    else:
        it = ACDgen(config,output_meshes=False).__iter__()
        if args.seed is not None:
            set_seed(args.seed)
        points, values_gt = next(it)    

        

    if args.gt:
        distances = values_gt
    else:
        distances = mark_cuts(points, args.checkpoint, config, args.no_threshold)

    lib_points = lib_neural_acd.VecArray3d(points[distances==1][:,:3].tolist())

    clusters = lib_neural_acd.dbscan(lib_points, config.lib.dbscan.eps, config.lib.dbscan.min_pts)

    # mesh = trimesh.Trimesh(vertices=structure.vertices, faces=structure.triangles)
    # mesh.show()

    show_pcd(points, distances,clusters=clusters)
    # save_rotating_pcd_gif(points, distances, gif_path="pcd_rotation.gif", frames=36)

