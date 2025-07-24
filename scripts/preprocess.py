import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..","lib", "build"))

import lib_neural_acd
import trimesh
from utils.misc import get_lib_mesh, normalize_mesh

def preprocess_mesh(mesh, scale=1.0, level_set=0.0):
    normalize_mesh(mesh)
    mesh = get_lib_mesh(mesh)
    lib_neural_acd.preprocess(mesh, scale, level_set)
    return trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.triangles)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a mesh for Neural ACD.")
    parser.add_argument("mesh_path", type=str, help="Path to the mesh file.")
    parser.add_argument("--scale", type=float, default=50.0, help="Scale factor for the mesh.")
    parser.add_argument("--level_set", type=float, default=0.05, help="Level set value for the mesh.")

    args = parser.parse_args()

    mesh = trimesh.load(args.mesh_path, force='mesh')

    mesh = preprocess_mesh(mesh, args.scale, args.level_set)

    mesh.show()
