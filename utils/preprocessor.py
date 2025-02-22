
import coacd_modified
import pyntcloud
import json
import multiprocessing
import threading
import trimesh
import torch

from utils.BaseUtils import *

coacd_modified.set_log_level("off")


BUFFER_SIZE = 64

WORKERS = 6

NUM_PLANES = 5

NUM_POINTS = 512


def recalculate_planes(cmesh):
    planes = coacd_modified.best_cutting_planes(cmesh,num_planes=NUM_PLANES)
    planes = [(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in planes]
    return planes


def process_mesh(mesh, plane_cache):
    cache_updated = False

    cmesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)
    result = coacd_modified.normalize(cmesh)

    normalized_mesh = trimesh.Trimesh(result.vertices, result.indices)

    rotation = apply_random_rotation(normalized_mesh)

    mesh_hash = str(hash((mesh.vertices, mesh.faces)))
    


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
        #try again
        planes = recalculate_planes(cmesh)
        plane_cache[mesh_hash] = planes
        cache_updated = True

        try:
            target = []
            for plane in planes:
                a, b, c, d = apply_rotation_to_plane(*plane[:4], rotation)
                target.append([a, b, c, d])
        except:
            mesh.export("broken_mesh.obj")
            return None, None, None
        
    if len(target) != NUM_PLANES:
        mesh.export("broken_mesh.obj")
        return None, None, None


    
    normalized_mesh.export(os.path.join("tmp", f"{str(hash((mesh.vertices, mesh.faces)))}.ply"), vertex_normal=True)
    pc_mesh = pyntcloud.PyntCloud.from_file(os.path.join("tmp", f"{str(hash((mesh.vertices, mesh.faces)))}.ply"))

    os.remove(os.path.join("tmp", f"{str(hash((mesh.vertices, mesh.faces)))}.ply"))
    pc = pc_mesh.get_sample("mesh_random", n=NUM_POINTS, normals=False, as_PyntCloud=True)



    return pc.points, target, cache_updated



def preprocess_data(loader, plane_cache,batch_size=16):

    batch_queue = queue.Queue(maxsize=BUFFER_SIZE)

    def preprocessor():
        cur_batch = ([],[])

        cache_updated = False

        for mesh in loader:
            
            points_, planes, updated = process_mesh(mesh, plane_cache)
            cache_updated = cache_updated or updated
            if points_ is not None:
                cur_batch[0].append(points_)
                cur_batch[1].append(planes)
            else:
                print("Skipping broken data")
                continue 
            
            if len(cur_batch[0]) == batch_size:

                points = np.array(cur_batch[0])
                points = torch.Tensor(points)
                points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
                points = points.float().cuda()

                target = np.array(cur_batch[1])
                target = torch.Tensor(target)
                target = target.float().cuda()

                def write_to_cache(plane_cache):
                    with open("plane_cache.json", "w") as plane_cache_f:
                        json.dump(plane_cache, plane_cache_f)

                if cache_updated:
                    t = threading.Thread(target=write_to_cache, args=(plane_cache,)) #prevents KeyboardInterrupt during write

                    t.start()
                    t.join()
                cache_updated = False
                cur_batch = ([],[])

                batch_queue.put((points, target))

        batch_queue.put(None)

    thread = threading.Thread(target=preprocessor, daemon=True)
    thread.start()

    while True:
        batch = batch_queue.get()
        if batch is None:
            break
        yield batch


# def worker_manager(loader, plane_cache,processed_queue):
#     executor = ProcessPoolExecutor()
#     futures = []
#     for _ in range(WORKERS):
#         mesh = next(loader, None)
#         if mesh is None:
#             processed_queue.put(None)
#             return
#         future = executor.submit(process_mesh, mesh, plane_cache)
#         futures.append(future)
    
#     while True:
#         for future in futures:
#             if future.done():
#                 points, target, updated = future.result()
#                 processed_queue.put((points, target))
#                 if updated:
#                     with open("plane_cache.json", "w") as plane_cache_f:
#                         json.dump(plane_cache, plane_cache_f)
#                 futures.remove(future)

#                 mesh = next(loader, None)
#                 if mesh is None:
#                     processed_queue.put(None)
#                     return
#                 new_future = executor.submit(process_mesh, mesh, plane_cache)
#                 futures.append(new_future)
#                 break
#         if len(futures) == 0:
#             return


# def preprocess_data(loader, plane_cache,batch_size=16):

#     processed_queue = queue.Queue(maxsize=BUFFER_SIZE)

#     worker_manager_thread = threading.Thread(target=worker_manager, args=(loader, plane_cache,processed_queue))
#     worker_manager_thread.start()

#     while True:
#         batch = ([], [])

#         for _ in range(batch_size):
#             processed = processed_queue.get()
#             if processed is None:
#                 return
#             batch[0].append(processed[0])
#             batch[1].append(processed[1])
        

#         points = np.array(batch[0])
#         points = torch.Tensor(points)
#         points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
#         points = points.float().cuda()

#         target = np.array(batch[1])
#         target = torch.Tensor(target)
#         target = target.float().cuda()

#         yield (points, target)