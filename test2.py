import coacd_modified
import trimesh

# Some of the code is taken from https://github.com/yanx27/Pointnet_Pointnet2_pytorch/tree/master CREDIT: Benny

import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import provider
import numpy as np
import coacd_modified
import pyntcloud
import json

from concurrent.futures import ThreadPoolExecutor, as_completed, wait

from pathlib import Path
from tqdm import tqdm
from utils.ShapeNetDataLoader import PartNormalDataset
from utils.BaseUtils import *
import threading

data_folder = "data/ShapeNetRedistributed"

tmp_folder = "tmp2"

for compressed in os.listdir(data_folder):
        if compressed.endswith(".zip"):
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
                            mesh_hash = str(hash((mesh.vertices, mesh.faces)))
                            if mesh_hash == "-6434849133867754989":
                                print(root,file)
                            
