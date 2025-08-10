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
from utils.visualization import save_rotating_pcd_gif

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

def show_pcd(points, distances):
    distances = np.asarray(distances, dtype=np.float32)
    points = points[:, :3]
    norm = plt.Normalize(vmin=np.min(distances), vmax=np.max(distances))
    base_colors = plt.get_cmap("jet")(norm(distances))[:, :3]

    # Blend positive distances with orange
    orange = np.array([1.0, 0.5, 0.0])
    brightness = np.clip(distances, 0, 1)[:, None]
    colors = base_colors * (1 - 0.5 * brightness) + orange * (0.5 * brightness)

    # Set alpha based on distance
    alpha = np.where(distances > 0, 1, 0.5)[:, None]
    rgba = np.hstack([colors, alpha])

    trimesh.points.PointCloud(points, colors=np.clip(rgba, 0, 1)).show()


def get_curvature(mesh, points, radius):
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)

    # for _ in range(5):
    #     vertices,triangles =  trimesh.remesh.subdivide(vertices, triangles)

    tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles)

    # tmesh.show()
    curvature = trimesh.curvature.discrete_gaussian_curvature_measure(tmesh, points, radius)

    curvature[curvature >= -0.1] = 0
    curvature[curvature < -0.1] = 1
    return curvature


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

        distances = []

        for start in range(0, points.shape[0], config.model.batch_size):
            end = min(start + config.model.batch_size, points.shape[0])
            batch_points = points[start:end]
    
            pred = model(batch_points, apply_sigmoid=True)

            distances.append(pred)
        distances = torch.cat(distances, dim=0)

        

        distances = distances.cpu().numpy()

    if no_threshold:
        if not batched:
            distances = distances.squeeze(0)
        return distances

    for i in range(len(distances)):
        distances[i][distances[i] < config.general.cut_point_threshold] = 0

        if len(np.flatnonzero(distances[i])) > config.general.cut_point_limit:
            # Keep only the top cut points based on distances
            # selected = np.argpartition(distances[i], -config.general.cut_point_limit)[-config.general.cut_point_limit:]

            nonzero_indices = np.flatnonzero(distances[i])

            selected = np.random.choice(nonzero_indices, size=config.general.cut_point_limit, replace=False)


            distances[i][selected] = 1
            distances[i][distances[i] < 1] = 0
        else:
            # If fewer than cut_point_limit points, set all to 1
            distances[i][distances[i] > 0] = 1

    if not batched:
        distances = distances.squeeze(0)

    return distances

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
        structure = trimesh.load(args.path, force='mesh')
        structure = get_lib_mesh(structure)
        normalize_mesh(structure)
        lib_neural_acd.preprocess(structure, 50.0, 0.55)

        points = lib_neural_acd.VecArray3d()
        point_tris = lib_neural_acd.VecInt()
        structure.extract_point_set(points, point_tris, config.general.num_points)

        

        points = np.asarray(points)


        tmesh = trimesh.Trimesh(vertices=np.asarray(structure.vertices), faces=np.asarray(structure.triangles))
        curvature = trimesh.curvature.discrete_gaussian_curvature_measure(tmesh, points, radius=0.02)
        normals = tmesh.face_normals[np.asarray(point_tris)]

        d = curvature.copy()
        d[d >= -0.0] = 0
        d[d < -0.0] = 1
        show_pcd(points, d)
        exit()


        points = np.hstack((points, curvature[:, np.newaxis],normals))

    else:
        it = ACDgen(config,output_meshes=False).__iter__()
        if args.seed is not None:
            set_seed(args.seed)
        points, distances_t = next(it)    

        


    # print(distances_t)
    if args.gt:
        distances = distances_t
    else:
        distances = mark_cuts(points, args.checkpoint, config, args.no_threshold )

    # mesh = trimesh.Trimesh(vertices=structure.vertices, faces=structure.triangles)
    # mesh.show()

    show_pcd(points, distances)
    # save_rotating_pcd_gif(points, distances, gif_path="pcd_rotation.gif", frames=36)

