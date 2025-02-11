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
    normal = np.array([a, c, -b]) #axis swap


    rotation = rotation[:3,:3]
    
    rotated_normal = rotation @ normal

    if np.linalg.norm(normal) == 0:
        raise ValueError("Invalid plane normal (0,0,0).")

    point_on_plane = -d * normal / np.linalg.norm(normal) ** 2 
    rotated_point = rotation @ point_on_plane 

    d_new = -np.dot(rotated_normal, rotated_point)

    return rotated_normal[0], -rotated_normal[2], rotated_normal[1], d_new #with axis swap


def load_shapenet(debug=False, batch_size=1):
    print("Loading ShapeNet dataset...")

    batch = []
    
    if debug and os.path.isdir("tmp"):
        #reuse already extracted data
        print("Using existing data in tmp folder")
        for root,dirs,files in os.walk("tmp"):
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
                    batch.append(mesh)
                    if len(batch) == batch_size:
                        yield batch
                        batch = []


    for compressed in os.listdir(os.path.join("data","ShapeNetCore")):
        if compressed.endswith(".zip"):
            empty_tmp()
            with zipfile.ZipFile(os.path.join("data","ShapeNetCore",compressed), 'r') as zip_ref:
                print(f"Extracting {compressed}...")
                zip_ref.extractall("tmp")
                print(f"Extracted {compressed}")
                for root,dirs,files in os.walk("tmp"):
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
                            batch.append(mesh)
                            if len(batch) == batch_size:
                                yield batch
                                batch = []
    empty_tmp()



def empty_tmp():
    print("Emptying tmp folder...")
    shutil.rmtree("tmp", ignore_errors=True)
            