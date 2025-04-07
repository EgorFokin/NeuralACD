from utils.BaseUtils import *
import coacd_modified
import numpy as np
import open3d as o3d

NUM_POINTS = 512

def process_mesh(mesh):

    cmesh = coacd_modified.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    _,result = coacd_modified.normalize(cmesh)

    normalized_mesh = o3d.geometry.TriangleMesh()
    normalized_mesh.vertices = o3d.utility.Vector3dVector(result.vertices)
    normalized_mesh.triangles = o3d.utility.Vector3iVector(result.indices)
    try:
        pcd = normalized_mesh.sample_points_poisson_disk(number_of_points=NUM_POINTS, init_factor=3)
    except:
        o3d.io.write_triangle_mesh("broken_mesh.obj", mesh)
        return None, None, None
    pc_points = np.asarray(pcd.points)
    
    return pc_points

def convert(data_folder,out):
    train_loader = load_shapenet(data_folder=data_folder)
    if not os.path.exists(out):
        os.makedirs(out)
    for mesh_hash,mesh in train_loader:
        points = process_mesh(mesh)
        with open(os.path.join(out,mesh_hash+".npy"),"wb") as f:
            np.save(f,points)

if __name__ == "__main__":
    convert("data/ShapeNetParts","data/ShapeNetPartsCloud")