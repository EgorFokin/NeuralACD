import coacd_modified
import open3d as o3d
import json
import threading
import torch
import numpy as np
import queue

from utils.BaseUtils import *

coacd_modified.set_log_level("off")

BUFFER_SIZE = 4
WORKERS = 6
NUM_PLANES = 5
NUM_POINTS = 512

def apply_random_rotation(mesh):
    rotation = o3d.geometry.get_rotation_matrix_from_xyz(np.random.rand(3) * 2 * np.pi)
    mesh.rotate(rotation, center=(0, 0, 0))
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

    if d_new < 0: #make the signs of coeffs consistent
        rotated_normal = -rotated_normal
        d_new = -d_new

    return rotated_normal[0], rotated_normal[1], rotated_normal[2], d_new 

def recalculate_planes(cmesh):
    planes = coacd_modified.best_cutting_planes(cmesh, num_planes=NUM_PLANES)
    planes = [(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in planes]
    return planes

def process_mesh(mesh_hash, mesh, plane_cache):
    cache_updated = False

    cmesh = coacd_modified.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
    result = coacd_modified.normalize(cmesh)

    normalized_mesh = o3d.geometry.TriangleMesh()
    normalized_mesh.vertices = o3d.utility.Vector3dVector(result.vertices)
    normalized_mesh.triangles = o3d.utility.Vector3iVector(result.indices)
    
    
    rotation = apply_random_rotation(normalized_mesh)
    
    if mesh_hash in plane_cache:
        planes = plane_cache[mesh_hash]
    else:
        print(f"Mesh {mesh_hash} not found in cache")
        planes = recalculate_planes(cmesh)
        plane_cache[mesh_hash] = planes
        cache_updated = True

    if len(planes) != NUM_PLANES:
        planes = recalculate_planes(cmesh)
        plane_cache[mesh_hash] = planes
        cache_updated = True
    
    try:
        target = []
        for plane in planes:
            a, b, c, d = apply_rotation_to_plane(*plane[:4], rotation)
            target.append([a, b, c, d])
    except Exception as e:
        print(e)
        planes = recalculate_planes(cmesh)
        plane_cache[mesh_hash] = planes
        cache_updated = True
        
        try:
            target = []
            for plane in planes:
                a, b, c, d = apply_rotation_to_plane(*plane[:4], rotation)
                target.append([a, b, c, d])
        except:
            o3d.io.write_triangle_mesh("broken_mesh.obj", mesh)
            return None, None, None
    
    if len(target) != NUM_PLANES:
        o3d.io.write_triangle_mesh("broken_mesh.obj", mesh)
        return None, None, None
    try:
        pcd = normalized_mesh.sample_points_poisson_disk(number_of_points=NUM_POINTS, init_factor=2)
    except:
        o3d.io.write_triangle_mesh("broken_mesh.obj", mesh)
        return None, None, None
    pc_points = np.asarray(pcd.points)
    
    del normalized_mesh, pcd, mesh
    return pc_points, target, cache_updated

def preprocess_data(loader, plane_cache, batch_size=16):
    batch_queue = queue.Queue(maxsize=BUFFER_SIZE)
    
    def preprocessor():
        cur_batch = ([], [])
        cache_updated = False
        
        for mesh_hash,mesh in loader:
            points_, planes, updated = process_mesh(mesh_hash,mesh, plane_cache)
            cache_updated = cache_updated or updated
            
            if points_ is not None:
                cur_batch[0].append(points_)
                cur_batch[1].append(planes)
            else:
                print("Skipping broken data")
                continue
            
            if len(cur_batch[0]) == batch_size:
                points = torch.tensor(np.array(cur_batch[0]), dtype=torch.float32).cuda()
                target = torch.tensor(np.array(cur_batch[1]), dtype=torch.float32).cuda()
                
                if cache_updated:
                    def write_to_cache():
                        with open("plane_cache.json", "w") as plane_cache_f:
                            json.dump(plane_cache, plane_cache_f)
                    
                    t = threading.Thread(target=write_to_cache)
                    t.start()
                    t.join()
                    cache_updated = False
                
                cur_batch = ([], [])

                batch_queue.put((points, target))
        
        batch_queue.put(None)
    
    thread = threading.Thread(target=preprocessor, daemon=True)
    thread.start()
    
    while True:
        batch = batch_queue.get()
        if batch is None:
            break
        yield batch

    
