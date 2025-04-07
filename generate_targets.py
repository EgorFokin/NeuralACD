import argparse
import os
import datetime
import numpy as np
import coacd_modified
import json

from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from pathlib import Path
from utils.BaseUtils import *
import threading


dict_lock = threading.Lock()



def process_mesh(mesh_hash,mesh, plane_cache):

    cmesh = coacd_modified.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))

    if mesh_hash not in plane_cache or len(plane_cache[mesh_hash]) != 5 or "e-3" in str(plane_cache[mesh_hash]):

        planes = coacd_modified.best_cutting_planes(cmesh, num_planes=5)
        if "e-3" in str(planes[0].score): # Weird bug where everything comes out as 1e-301
            planes = coacd_modified.best_cutting_planes(cmesh, num_planes=5)

        planes = [(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in planes]
        print(mesh_hash,planes)
        with dict_lock:
            plane_cache[mesh_hash] = planes

def future_done(future):
    global threads_running, meshes_processed
    threads_running -= 1
    meshes_processed += 1
    print(f"Processed {meshes_processed} meshes")


        

NUM_THREADS = 30

if __name__ == "__main__":
    global threads_running, meshes_processed
    threads_running = 0
    meshes_processed = 0
    i = 0
    coacd_modified.set_log_level("off")
    start_time = datetime.datetime.now()
    time_spent = datetime.timedelta()
    ten_prev = []
    with open("plane_cache.json", "r") as plane_cache_f:
        plane_cache = json.load(plane_cache_f)


    loader = load_shapenet(debug=False, data_folder="data/ShapeNetParts")
    executor = ThreadPoolExecutor()
    futures = []
    while True:
        if threads_running < NUM_THREADS:
            mesh_hash,mesh = next(loader,None)
            if mesh is None:
                break
            threads_running += 1
            
            future = executor.submit(process_mesh,mesh_hash, mesh, plane_cache)
            future.add_done_callback(future_done)
            futures.append(future)
        
            if meshes_processed % 50 == 0:
                with open("tmp.json", "w") as plane_cache_f:
                    with dict_lock:
                        json.dump(plane_cache, plane_cache_f)
                    
                os.replace("tmp.json", "plane_cache.json")


    wait(futures)
    with open("tmp.json", "w") as plane_cache_f:
        with dict_lock:
            json.dump(plane_cache, plane_cache_f)
                    
    os.replace("tmp.json", "plane_cache.json")