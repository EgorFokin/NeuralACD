import torch
import open3d as o3d
import coacd_modified
import numpy as np
from model import model
import random

MODEL_PATH =  "C:\\Users\\egorf\\Desktop\\cmpt469\\DeepConvexDecomposition\\log\\2025-03-14_11-55\\checkpoints\\checkpoint.pth"
MESH_PATH = "cow-nonormals.obj"
np.random.seed(0)

predictor = model.get_model(4).cuda()

predictor.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])

def revert_normalization(mesh,bbox):
    x_min, x_max, y_min, y_max, z_min, z_max = bbox

    m_len = max(x_max - x_min, y_max - y_min, z_max - z_min)
    m_Xmid = (x_max + x_min) / 2
    m_Ymid = (y_max + y_min) / 2
    m_Zmid = (z_max + z_min) / 2

    vertices = np.asarray(mesh.vertices)

    # Scale and translate vertices
    vertices = vertices / 2 * m_len + np.array([m_Xmid, m_Ymid, m_Zmid])

    # Update mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals() 

def split(mesh):


    cmesh = coacd_modified.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    bbox,normalized_cmesh = coacd_modified.normalize(cmesh)


    normalized_mesh = o3d.geometry.TriangleMesh()
    normalized_mesh.vertices = o3d.utility.Vector3dVector(normalized_cmesh.vertices)
    normalized_mesh.triangles = o3d.utility.Vector3iVector(normalized_cmesh.indices)

    pnt = normalized_mesh.sample_points_poisson_disk(number_of_points=512, init_factor=3)
    points = np.asarray(pnt.points)

    points = points.reshape(1, -1, 512)
    points = torch.tensor(points, dtype=torch.float32).cuda()

    with torch.no_grad():
        plane = predictor(points)
    
    plane = coacd_modified.CoACD_Plane(*list(plane.cpu().numpy()[0]),0)
    result = coacd_modified.clip(normalized_cmesh, plane)

    parts = []

    for vs,fs in result:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vs)
        mesh.triangles = o3d.utility.Vector3iVector(fs)
        revert_normalization(mesh,bbox)
        parts.append(mesh)

    return parts



    

def decompose(mesh,depth):
    if depth == 0:
        return [mesh]
    
    parts = split(mesh)

    result = []
    for part in parts:
        result += decompose(part,depth-1)
    
    return result
    

if __name__ == "__main__":
    DEPTH = 3
    mesh = o3d.io.read_triangle_mesh(MESH_PATH)



    parts = decompose(mesh,DEPTH)

    for i,part in enumerate(parts):
        part.paint_uniform_color([random.random(),random.random(),random.random()])
    
    o3d.visualization.draw_geometries(parts)
        



