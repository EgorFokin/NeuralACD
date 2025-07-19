import sys

sys.path.append("lib/build")


import coacd
import trimesh
import numpy as np
import lib_acd_gen
from utils.ACDgen import ACDgen
import os

NUM_SAMPLES = 100


if os.path.exists("stats.txt"):
    print("Stats file already exists. Please remove it before running the script.")
    exit()


it = ACDgen(output_meshes=True).__iter__()
lib_acd_gen.set_seed(44)
#next(it)  

for i in range(NUM_SAMPLES):
    points, distances_t, structure = next(it)

    mesh = coacd.Mesh(np.asarray(structure.vertices), np.asarray(structure.triangles))
    result = coacd.run_coacd(mesh)

    # mesh_parts = []
    # for vs, fs in result:
    #     mesh_parts.append(trimesh.Trimesh(vs, fs))

    # scene = trimesh.Scene()
    # np.random.seed(0)
    # for p in mesh_parts:
    #     p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
    #     scene.add_geometry(p)
    # scene.export("decomposed.obj")

    # break


total_concavity = 0.0
total_parts = 0

with open("stats.txt", "r") as f:
    for line in f.readlines():
        concavity, num_parts = map(float, line.strip().split(';'))
        total_concavity += concavity
        total_parts += num_parts
print(f"Average Concavity: {total_concavity / NUM_SAMPLES}")
print(f"Average Parts: {total_parts / NUM_SAMPLES}")