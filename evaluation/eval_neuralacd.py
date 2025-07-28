import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..","lib", "build"))

import numpy as np
import torch
import sys
import os
import tempfile
import argparse
import multiprocessing as mp

import lib_neural_acd

from utils.VHACD import VHACD
from utils.ACDgen import ACDgen
from utils.misc import *
from scripts.mark_cuts import mark_cuts
from scripts.decompose import decompose

def process_sample(points, structure):

    mesh = lib_neural_acd.Mesh()
    vertices_vector = lib_neural_acd.VecArray3d(structure[0])
    triangles_vector = lib_neural_acd.make_vecarray3i(structure[1])


    mesh.vertices = vertices_vector
    mesh.triangles = triangles_vector
    


    with tempfile.NamedTemporaryFile(mode='r+') as temp_file:
        res = decompose(mesh, points, temp_file.name)
        results = temp_file.readlines()[0]
        concavity, num_parts = map(float, results.strip().split(';'))
        print(f"Concavity: {concavity}, Parts: {num_parts}")
        return concavity, num_parts


def evaluate(checkpoint, config, num_samples, num_workers=1, is_vhacd=False):
    points = []
    structures = []


    if is_vhacd:
        dataset = VHACD()
        num_samples = len(dataset)
        for i in range(len(dataset)):
            p, st = dataset[i]
            points.append(p)
            structures.append([np.asarray(st.vertices), np.asarray(st.triangles)])
        
        points = torch.stack(points, dim=0)
    else:

        it = ACDgen(config,output_meshes=True).__iter__()


        # full_structures = []
        for i in range(num_samples):
            p, _, st = next(it)

            # full_structures.append(st)

            structures.append([np.asarray(st.vertices), np.asarray(st.triangles)])
            points.append(p)
        


        points = np.asarray(points)
    

    distances = mark_cuts(points, checkpoint, config)
    # distances = []
    # for i in range(len(points)):
    #     distances.append(get_curvature(full_structures[i], points[i], radius=0.02))

    cut_points = [p[distances[i] == 1] for i, p in enumerate(points)]


    # show_pcd(points[2], distances[2])
    # exit()

    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            results = pool.starmap(process_sample, zip(cut_points, structures))
    else:
        results = []
        for p, st in zip(cut_points, structures):
            metrics = process_sample(p, st)
            results.append(metrics)

    total_concavity = sum(res[0] for res in results)
    total_parts = sum(res[1] for res in results)
    print(f"Average Concavity: {total_concavity / num_samples}")
    print(f"Average Parts: {total_parts / num_samples}")


if __name__ == "__main__":



    parser = argparse.ArgumentParser(description="Evaluate NeuralACD model.")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoint.ckpt", help="Path to the model checkpoint.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file.")
    parser.add_argument("--num-samples", type=int, default=100, help="Number of samples to evaluate.")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--num-workers", type=int, default=1, help="Number of parallel workers for evaluation.")
    parser.add_argument("--vhacd", action="store_true", help="Use VHACD dataset.")

    args = parser.parse_args()

    config = load_config(args.config)
    lib_neural_acd.config.process_output_parts = True

    if args.seed is not None:
        set_seed(args.seed)


    evaluate(args.checkpoint, config, args.num_samples, args.num_workers, args.vhacd)