import lib_acd_gen
from model.model import ACDModel
import open3d as o3d
import numpy as np
from matplotlib import cm
import trimesh
import torch


# structure = lib_acd_gen.generate_sphere_structure(10)

# verts = []
# triangles = []
# cut_verts = []
# vertex_offset = 0
# for mesh in structure:
#     verts.extend(mesh.vertices)
#     triangles.extend([[v0 + vertex_offset, v1 + vertex_offset, v2 + vertex_offset] for v0, v1, v2 in mesh.triangles])

#     vertex_offset += len(mesh.vertices)
#     cut_verts.extend(mesh.cut_verts)

# o3d_mesh = o3d.geometry.TriangleMesh()

# #print(verts)
# o3d_mesh.vertices = o3d.utility.Vector3dVector(verts)
# o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)

o3d_mesh = o3d.io.read_triangle_mesh("data/meshes/cow.obj")

pcd = o3d_mesh.sample_points_uniformly(number_of_points=10000).points

points = torch.tensor(pcd, dtype=torch.float32).cuda()


with torch.no_grad():
    model = ACDModel().cuda()
    model.load_state_dict(torch.load("checkpoints/08,07,2025-18:10:08/best-model-train_loss=0.013116518966853619.ckpt")["state_dict"])
    model.eval()
    distances = model(points.unsqueeze(0)).squeeze().cpu().numpy()

scaled = distances*distances*2



# Map to color (e.g., blue → red gradient using matplotlib)
colormap = cm.get_cmap("jet")
colors = colormap(1.0 - scaled)[:, :3]  # RGB, invert to make close = red

alpha = 1.0 - np.clip(scaled, 0.0, 0.6)
rgba = np.hstack([colors, alpha[:, np.newaxis]])

point_cloud = trimesh.points.PointCloud(pcd, colors=rgba)

point_cloud.show()