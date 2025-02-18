# Some of the code is taken from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master CREDIT: Benny

import argparse
import os
import datetime
import logging
import sys
import importlib
import shutil
import numpy as np
import coacd_modified
import json

from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from pathlib import Path
from utils.BaseUtils import *
import threading

file_lock = threading.Lock()


def process_mesh(mesh, plane_cache):
    cmesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)

    mesh_hash = str(hash((mesh.vertices, mesh.faces)))
    if mesh_hash not in plane_cache:

        planes = coacd_modified.best_cutting_planes(cmesh, num_planes=5)

        planes = [(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in planes]
        print(str(hash((mesh.vertices, mesh.faces))),planes)
        plane_cache[mesh_hash] = planes

def preprocess_data(batch):
    with open("plane_cache.json", "r") as plane_cache_f:
        plane_cache = json.load(plane_cache_f)

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_mesh, mesh, plane_cache) for mesh in batch]
    wait(futures)

    with file_lock:
        with open("plane_cache.json", "w") as plane_cache_f:
            json.dump(plane_cache, plane_cache_f)

TOTAL_MESHES = 52000

if __name__ == "__main__":
    i = 0
    coacd_modified.set_log_level("off")
    batch_size = 30
    start_time = datetime.datetime.now()
    time_spent = datetime.timedelta()
    ten_prev = []
    for batch in load_shapenet(debug=False, batch_size=batch_size, data_folder="data/ShapenetRedistributed"):
        preprocess_data(batch)
        i+=1
        delta = datetime.datetime.now() - start_time
        #print remaining time
        ten_prev.append(delta)
        time_spent += delta
        if i>10:
            print(f"Mesh {i*batch_size}/{TOTAL_MESHES}; {str(time_spent).split('.')[0]}/{str(sum(ten_prev, datetime.timedelta())/10*(TOTAL_MESHES/(batch_size))).split('.')[0]}")
            ten_prev.pop(0)
        else:
            print(f"Mesh {i*batch_size}/{TOTAL_MESHES}; {str(time_spent).split('.')[0]}/{str(delta*(TOTAL_MESHES/(batch_size))).split('.')[0]}")
        start_time = datetime.datetime.now()