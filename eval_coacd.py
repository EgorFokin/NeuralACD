import sys

sys.path.append("lib/build")


import coacd
import trimesh
import numpy as np
import lib_acd_gen
from utils.ACDgen import ACDgen
import os

NUM_SAMPLES = 100


# if os.path.exists("stats.txt"):
#     print("Stats file already exists. Please remove it before running the script.")
#     exit()


# it = ACDgen(output_meshes=True).__iter__()
# lib_acd_gen.set_seed(42)
# #next(it)  

# for i in range(NUM_SAMPLES):
#     points, distances_t, structure = next(it)

#     mesh = coacd.Mesh(np.asarray(structure.vertices), np.asarray(structure.triangles))
#     result = coacd.run_coacd(mesh)


total_concavity = 0.0
total_parts = 0

with open("stats.txt", "r") as f:
    for line in f.readlines():
        concavity, num_parts = map(float, line.strip().split(';'))
        total_concavity += concavity
        total_parts += num_parts
print(f"Average Concavity: {total_concavity / NUM_SAMPLES}")
print(f"Average Parts: {total_parts / NUM_SAMPLES}")