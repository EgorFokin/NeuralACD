# Some of the code is taken from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master CREDIT: Benny

import argparse
import os
import datetime
import sys
import importlib
import shutil
import numpy as np
import coacd_modified
import json

from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from pathlib import Path
from BaseUtils import *
import threading

file_lock = threading.Lock()


def process_mesh(mesh, plane_cache):
    cmesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)

    mesh_hash = str(hash((mesh.vertices, mesh.faces)))

    if mesh_hash not in plane_cache:

        print(mesh_hash)
        planes = coacd_modified.best_cutting_planes(cmesh, num_planes=5)

        planes = [(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in planes]
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


if __name__ == "__main__":
    i = 0
    coacd_modified.set_log_level("off")
    batch_size = 30
    for batch in load_shapenet(batch_size=batch_size):
        preprocess_data(batch)
        i+=1
        print(f"Processed {i * batch_size} meshes")
