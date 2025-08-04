import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..","lib", "build"))

import lib_neural_acd
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
from scripts.mark_cuts import mark_cuts, show_pcd

def get_planes(cut_points):
    if len(cut_points) == 0:
        return []
    cut_points = lib_neural_acd.VecArray3d(cut_points.tolist())
    planes = lib_neural_acd.get_best_planes(cut_points)

    return list(planes)


def get_trimesh_plane(a, b, c, d):
    # Normalize the normal
    normal = np.array([a, b, c], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    # Find a point on the plane that's inside or near the [0,1]^3 box
    # We'll use the projection of the center of the box (0.5, 0.5, 0.5) onto the plane
    center = np.array([0.5, 0.5, 0.5])
    distance = np.dot(normal, center) + d
    point_on_plane = center - distance * normal

    # Create a small quad (plane proxy) within [0,1]^3
    size = 4  # slightly larger than the box to fill view
    plane = trimesh.creation.box(extents=(size, size, 0.005))  # very thin
    plane.apply_translation(-plane.centroid)

    # Rotate plane to match normal
    T = trimesh.geometry.align_vectors([0, 0, 1], normal)
    if T is not None:
        plane.apply_transform(T)

    # Translate to the projected point
    plane.apply_translation(point_on_plane)

    # Set visual style
    plane.visual.face_colors = [255, 0, 0, 100]  # red, semi-transparent
    return plane

def visualize(planes,points, distances):

    scene = trimesh.Scene()

    for plane in planes:
        trimesh_plane = get_trimesh_plane(plane[0], plane[1], plane[2], plane[3])
        print(plane[0], plane[1], plane[2], plane[3])
        scene.add_geometry(trimesh_plane)

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

    scene.add_geometry(trimesh.points.PointCloud(points, colors=np.clip(rgba, 0, 1)))

    scene.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mark cuts in point cloud using ACD model.")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoint.ckpt", help="Path to the model checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--gt", action="store_true", help="Use ground truth points instead of model predictions.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file.")
    parser.add_argument("--path", type=str, default="", help="Path to the mesh file.")
    parser.add_argument("--no-threshold", action="store_true", help="Do not apply thresholding to cut points.")
    
    args = parser.parse_args()
    
    config = load_config(args.config)

    if args.path:
        structure = trimesh.load(args.path, force='mesh')
        structure = get_lib_mesh(structure)
        normalize_mesh(structure)
        lib_neural_acd.preprocess(structure, 50.0, 0.05)

        points = lib_neural_acd.VecArray3d()
        point_tris = lib_neural_acd.VecInt()
        structure.extract_point_set(points, point_tris, config.general.num_points)

        points = np.asarray(points)


        tmesh = trimesh.Trimesh(vertices=np.asarray(structure.vertices), faces=np.asarray(structure.triangles))
        tmesh.export('test.ply')
        curvature = trimesh.curvature.discrete_gaussian_curvature_measure(tmesh, points, radius=0.02)
        normals = tmesh.face_normals[np.asarray(point_tris)]
        points = np.hstack((points, curvature[:, np.newaxis],normals))
    else:
        it = ACDgen(config,output_meshes=True).__iter__()
        if args.seed is not None:
            set_seed(args.seed)
        points, distances_t, structure = next(it)    


    # print(distances_t)
    if args.gt:
        distances = distances_t
    else:
        distances = mark_cuts(points, args.checkpoint, config, args.no_threshold )

    # mesh = trimesh.Trimesh(vertices=structure.vertices, faces=structure.triangles)
    # mesh.show()

    points = np.asarray(points)
    points = points[:, :3]  # Ensure points are 3D

    cut_points = points[distances == 1]
    print(f"Number of cut points: {len(cut_points)}")
    planes = get_planes(cut_points)

    visualize(planes, points, distances)

