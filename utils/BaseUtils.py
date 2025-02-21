import trimesh
import os
import zipfile
import shutil
import numpy as np

def apply_random_rotation(mesh):
    rotation = trimesh.transformations.random_rotation_matrix()


    #rotation = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1]])

    mesh.apply_transform(rotation)
    return rotation

def apply_rotation_to_plane(a,b,c,d,rotation):
    normal = np.array([a, b, c])


    rotation = rotation[:3,:3]
    
    rotated_normal = rotation @ normal

    if np.linalg.norm(normal) == 0:
        raise ValueError("Invalid plane normal (0,0,0).")

    point_on_plane = -d * normal / np.linalg.norm(normal) ** 2 
    rotated_point = rotation @ point_on_plane 

    d_new = -np.dot(rotated_normal, rotated_point)

    if rotated_normal[0] < 0: #make the signs of coeffs consistent
        rotated_normal = -rotated_normal
        d_new = -d_new

    return rotated_normal[0], rotated_normal[1], rotated_normal[2], d_new 


def load_shapenet(debug=False, tmp_folder="tmp", data_folder="data/ShapeNetCore"):
    print("Loading ShapeNet dataset...")

    
    if debug and os.path.isdir(tmp_folder):
        #reuse already extracted data
        print("Using existing data in tmp folder")
        for root,dirs,files in os.walk(tmp_folder):
            for file in files:
                if file.endswith(".obj"):
                    obj = trimesh.load(os.path.join(root,file))
                    #check if the obj contains a scene instead of a single mesh
                    if isinstance(obj,trimesh.Scene):
                        if len(obj.geometry) == 0:
                            continue
                        else:
                            mesh = trimesh.util.concatenate(
                                tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                                    for g in obj.geometry.values()))
                    else:
                        mesh = obj
                    yield mesh


    for compressed in os.listdir(data_folder):
        if compressed.endswith(".zip"):
            empty_tmp()
            with zipfile.ZipFile(os.path.join(data_folder,compressed), 'r') as zip_ref:
                print(f"Extracting {compressed}...")
                zip_ref.extractall(tmp_folder)
                print(f"Extracted {compressed}")
                for root,dirs,files in os.walk(tmp_folder):
                    for file in files:
                        if file.endswith(".obj"):
                            obj = trimesh.load(os.path.join(root,file))
                            #check if the obj contains a scene instead of a single mesh
                            if isinstance(obj,trimesh.Scene):
                                if len(obj.geometry) == 0:
                                    continue
                                else:
                                    mesh = trimesh.util.concatenate(
                                        tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
                                            for g in obj.geometry.values()))
                            else:
                                mesh = obj
                            yield mesh
    empty_tmp(tmp_folder)



def empty_tmp(tmp_folder="tmp"):
    print("Emptying tmp folder...")
    shutil.rmtree(tmp_folder, ignore_errors=True)
            