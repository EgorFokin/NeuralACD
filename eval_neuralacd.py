
import open3d as o3d
import numpy as np
from matplotlib import cm
import trimesh
import torch
import sys
import os
sys.path.append("lib/build")

import lib_acd_gen

from utils.jlinkage import JLinkage

from sklearn.cluster import DBSCAN

from utils.ACDgen import ACDgen

from model.model import ACDModel
from scipy.spatial import cKDTree

NUM_SAMPLES = 100

CHECKPOINT="checkpoints/13,07,2025-19:01:21/best-model-ema_loss=0.29254528880119324.ckpt"

def remove_outliers(points, threshold=0.05):
    # Remove points that have less than 3 neighbors within the threshold
    
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, threshold)
    inliers = [i for i, n in enumerate(neighbors) if len(n) >= 3]
    return points[inliers]

model = ACDModel().cuda()
model.load_state_dict(torch.load(CHECKPOINT)["state_dict"])
model.eval()

it = ACDgen(output_meshes=True).__iter__()
lib_acd_gen.set_seed(42)
#next(it)  

for i in range(NUM_SAMPLES):
    points, distances_t, structure = next(it)

    points = np.asarray(points)

    with torch.no_grad():
        distances = model(torch.tensor(points, dtype=torch.float32).cuda().unsqueeze(0))
        distances = torch.sigmoid(distances)  # Ensure distances are in [0, 1] range
        distances = distances.squeeze().cpu().numpy()

    threshold = 0.4
    distances[distances < threshold] = 0
    distances[distances >= threshold] = 1

    cut_points = points[distances == 1]
    cut_points = remove_outliers(cut_points, threshold=0.05)

    cut_points_vector = lib_acd_gen.VecArray3d(cut_points.tolist())
    decomposed = lib_acd_gen.process(structure,cut_points_vector)

    scene = trimesh.Scene()

    for mesh in decomposed:
        triangles = np.asarray(mesh.triangles)
        vertices = np.asarray(mesh.vertices)
        if (triangles.shape[0] == 0 or vertices.shape[0] == 0):
            continue
        tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=True)

        #apply a random color to each mesh
        tmesh.visual.face_colors = np.random.randint(0, 255, (3), dtype=np.uint8)

        scene.add_geometry(tmesh)

    scene.export("decomposed.glb")
    

    tmesh = trimesh.Trimesh(vertices=structure.vertices, faces=structure.triangles, process=True)
    tmesh.export("structure.glb")
    scene.show()

    break


total_concavity = 0.0
total_parts = 0

with open("stats.txt", "r") as f:
    for line in f.readlines():
        concavity, num_parts = map(float, line.strip().split(';'))
        total_concavity += concavity
        total_parts += num_parts
print(f"Average Concavity: {total_concavity / NUM_SAMPLES}")
print(f"Average Parts: {total_parts / NUM_SAMPLES}")