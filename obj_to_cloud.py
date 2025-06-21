from utils.BaseUtils import *
import coacd_modified
import numpy as np
import open3d as o3d

NUM_POINTS = 512

blacklist = ['908e85e13c6fbde0a1ca08763d503f0e','4708d67a361201b2ff7e95552a1d6a0e']

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
    for compressed in os.listdir(data_folder):
        if compressed.endswith(".zip"):
            empty_tmp()
            with zipfile.ZipFile(os.path.join(data_folder,compressed), 'r') as zip_ref:
                print(f"Extracting {compressed}...")
                zip_ref.extractall("tmp")
                print(f"Extracted {compressed}")
                for root,dirs,files in os.walk("tmp"):
                    for file in files:
                        if file.endswith(".obj"):
                            
                            mesh_hash = root.split(os.path.sep)[-2]

                            if mesh_hash+'.npy' in os.listdir(out) or  mesh_hash in blacklist:
                                continue
                            mesh = o3d.io.read_triangle_mesh(os.path.join(root,file))

                            points = process_mesh(mesh)
                            print(mesh_hash)
                            with open(os.path.join(out,mesh_hash+".npy"),"wb") as f:
                                np.save(f,points)

if __name__ == "__main__":
    convert("data/ShapeNetCore","data/ShapeNetPointCloud")