import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..","lib", "build"))

from scripts.mark_cuts import mark_cuts
import lib_neural_acd
from utils.ACDgen import ACDgen
import trimesh
import numpy as np
import argparse
from utils.misc import load_config, get_point_cloud, get_lib_mesh, normalize_mesh

def show_geometry(meshes, save_path=None):
    scene = trimesh.Scene()
    for mesh in meshes:
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        if (triangles.shape[0] == 0 or vertices.shape[0] == 0):
            continue
        tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
        tmesh.visual.face_colors = np.random.randint(0, 255, (3), dtype=np.uint8)
        scene.add_geometry(tmesh)
    scene.show()
    if save_path:
        scene.export(save_path)

def decompose(mesh, cut_points, stats_file=""):
    cut_points_vector = lib_neural_acd.VecArray3d(cut_points.tolist())
    result = lib_neural_acd.process(mesh, cut_points_vector,stats_file)

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decompose a mesh using ACD model.")
    parser.add_argument("--path", type=str, default="", help="Path to the mesh file.")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoint.ckpt", help="Path to the model checkpoint.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--gt", action="store_true", help="Use ground truth points instead of model predictions.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file.")
    
    args = parser.parse_args()

    config = load_config(args.config)
    
    if args.path:
        structure = trimesh.load(args.path, force='mesh')
        structure.show()
        structure = get_lib_mesh(structure)
        structure = normalize_mesh(structure)
        lib_neural_acd.preprocess(structure, 50.0, 0.05)

        pcd = get_point_cloud(structure)
        points = np.asarray(pcd.points)

    else:
        it = ACDgen(output_meshes=True).__iter__()
        if args.seed is not None:
            lib_neural_acd.set_seed(args.seed)
        points, distances_t, structure = next(it)    

        pcd = get_point_cloud(structure)

        points = np.asarray(pcd.points)

    if args.gt:
        distances = distances_t
    else:
        distances = mark_cuts(points, args.checkpoint, config)

    parts = decompose(structure, points[distances == 1])
    show_geometry(parts, save_path="decomposed.glb")


