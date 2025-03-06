import os
import zipfile
import shutil
import sys
import torch

import open3d as o3d


from concurrent.futures import ProcessPoolExecutor, as_completed




def load_shapenet(debug=False, tmp_folder="tmp", data_folder="data/ShapeNetCore"):
    print("Loading ShapeNet dataset...")

    if debug and os.path.isdir(tmp_folder):
        for root,dirs,files in os.walk(tmp_folder):
            for file in files:
                if file.endswith(".obj"):
                    mesh_hash = ''.join(file.split('.')[:-1])
                    mesh = o3d.io.read_triangle_mesh(os.path.join(root,file))
                    yield mesh_hash,mesh


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
                            mesh_hash = ''.join(file.split('.')[:-1])
                            mesh = o3d.io.read_triangle_mesh(os.path.join(root,file))
                            yield mesh_hash,mesh
    empty_tmp(tmp_folder)



# def load_mesh(file):
#     obj = trimesh.load(file)
#     #check if the obj contains a scene instead of a single mesh
#     if isinstance(obj,trimesh.Scene):
#         if len(obj.geometry) == 0:
#             return None
#         else:
#             mesh = trimesh.util.concatenate(
#                 tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
#                     for g in obj.geometry.values()))
#     else:
#         mesh = obj
#     return mesh


# def load_shapenet(debug=False, tmp_folder="tmp", data_folder="data/ShapeNetCore"):
#     print("Loading ShapeNet dataset...")

#     mesh_queue = queue.Queue()

#     num_processes = 0

#     def process_ended(future):
#         nonlocal num_processes
#         res = future.result()
#         if res is not None:
#             mesh_queue.put(res)
#         num_processes -= 1

#     def producer():
#         nonlocal num_processes
#         """Extracts and loads meshes into the mesh_queue asynchronously."""

#         executor = ProcessPoolExecutor(max_workers=4)

#         if debug and os.path.isdir(tmp_folder):
#             #reuse already extracted data
#             print("Using existing data in tmp folder")
#             for root,dirs,files in os.walk(tmp_folder):
#                 for file in files:
#                     if file.endswith(".obj"):
#                         while mesh_queue.qsize()+num_processes >= BUFFER_SIZE:
#                             pass

#                         future=executor.submit(load_mesh, os.path.join(root,file))
#                         future.add_done_callback(process_ended)
#                         num_processes += 1


#         for compressed in os.listdir(data_folder):
#             if compressed.endswith(".zip"):
#                 empty_tmp(tmp_folder)
#                 with zipfile.ZipFile(os.path.join(data_folder, compressed), 'r') as zip_ref:
#                     print(f"Extracting {compressed}...")
#                     zip_ref.extractall(tmp_folder)
#                     print(f"Extracted {compressed}")

#                     for root, dirs, files in os.walk(tmp_folder):
#                         for file in files:
#                             if file.endswith(".obj"):
                                
#                                 while mesh_queue.qsize()+num_processes >= BUFFER_SIZE:
#                                     time.sleep(0.1)
#                                     pass

#                                 future=executor.submit(load_mesh, os.path.join(root,file))
#                                 future.add_done_callback(process_ended)
#                                 num_processes += 1

#         mesh_queue.put(None)  # Sentinel value to signal completion

#     thread = threading.Thread(target=producer, daemon=True)
#     thread.start()

#     # Yield meshes from the mesh_queue
#     while True:
#         mesh = mesh_queue.get()
#         if mesh is None:
#             break
#         yield mesh
    
#     empty_tmp(tmp_folder)


def empty_tmp(tmp_folder="tmp"):
    print("Emptying tmp folder...")
    shutil.rmtree(tmp_folder, ignore_errors=True)
            