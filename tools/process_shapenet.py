import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.BaseUtils import *
import coacd_modified
import numpy as np
import open3d as o3d
import argparse
import h5py
from tqdm import tqdm
import zipfile
import warnings


coacd_modified.set_log_level("error")
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

NUM_POINTS = 512

blacklist = ['908e85e13c6fbde0a1ca08763d503f0e','4708d67a361201b2ff7e95552a1d6a0e','256b66cbb1bd1c88b2cb9dff9356b165']

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

def get_dataset_size(data_folder):
    total_meshes = 0
    prev = []
    for compressed in os.listdir(data_folder):
        if compressed.endswith(".zip"):
            with zipfile.ZipFile(os.path.join(data_folder,compressed), 'r') as zip_ref:
                for name in zip_ref.namelist():
                    if name.endswith(".obj"):
                        
                        if name.split(os.path.sep)[-3] not in prev:
                            total_meshes += 1
                        prev.append(name.split(os.path.sep)[-3])

    return total_meshes

def get_already_processed(output_folder):
    processed_meshes = set()
    for file_name in ['train_data.h5', 'val_data.h5']:
        file_path = os.path.join(output_folder, file_name)
        if os.path.exists(file_path):
            with h5py.File(file_path, 'r') as f:
                if 'hashes' in f:
                    processed_meshes.update([h.decode('utf-8') for h in f['hashes']])
    return processed_meshes

def update_h5_file(file_name, mesh_hash, points):
    with h5py.File(file_name, 'a') as f:
        if 'point_clouds' not in f:
            f.create_dataset('point_clouds', data=np.empty((0, NUM_POINTS, 3), dtype=np.float32), maxshape=(None, NUM_POINTS, 3))
        
        point_clouds = f['point_clouds']
        new_shape = (point_clouds.shape[0] + 1, NUM_POINTS, 3)
        point_clouds.resize(new_shape)
        point_clouds[-1] = points
        
        if 'hashes' not in f:
            f.create_dataset('hashes', data=np.empty((0,), dtype=h5py.string_dtype(encoding='utf-8')), maxshape=(None,))
        
        hashes = f['hashes']
        hashes.resize((hashes.shape[0] + 1,))
        hashes[-1] = mesh_hash.encode('utf-8')

def convert(data_folder,output_folder, validation_percentage=0):
    dataset_size = get_dataset_size(data_folder)


    processed = get_already_processed(output_folder)

    pbar = tqdm(total=dataset_size-len(processed), desc="Meshes processed", unit="mesh")

    for compressed in os.listdir(data_folder):
        if compressed.endswith(".zip"):
            empty_tmp()
            with zipfile.ZipFile(os.path.join(data_folder,compressed), 'r') as zip_ref:
                zip_ref.extractall("tmp")
                print(f"Extracted {compressed}")
                for root,dirs,files in os.walk("tmp"):
                    for file in files:
                        if file.endswith(".obj"):
                            
                            mesh_hash = root.split(os.path.sep)[-2]
                            if mesh_hash in blacklist:
                                print(f"Skipping blacklisted mesh: {mesh_hash}")
                                continue
                            elif mesh_hash in processed:
                                continue

                            mesh = o3d.io.read_triangle_mesh(os.path.join(root,file))
                            print(mesh_hash)

                            points = process_mesh(mesh)

                            file_name = 'train_data.h5'

                            if validation_percentage > 0 and np.random.rand() < validation_percentage / 100:
                                file_name = 'val_data.h5'
                            
                            update_h5_file(os.path.join(output_folder,file_name), mesh_hash, points)
                            processed.add(mesh_hash)
                            pbar.update(1)
    pbar.close()


                            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ShapeNet dataset to point clouds")
    parser.add_argument('--data_folder', type=str, default="data/ShapeNetCore", help='Path to the ShapeNet dataset folder')
    parser.add_argument('--output_folder', type=str, default="data", help='Path to save the point clouds')
    parser.add_argument('--validation_percentage', type=float, default=0, help='Percentage of data to use for validation')
    args = parser.parse_args()
    convert(args.data_folder, args.output_folder, args.validation_percentage)
    