import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..","lib", "build"))


import coacd
import numpy as np
from utils.VHACD import VHACD
from utils.ACDgen import ACDgen
import os
import argparse
from utils.misc import *


NUM_SAMPLES = 100


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate COACD model.")
    parser.add_argument("--vhacd", action="store_true", help="Use VHACD dataset.")
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES, help="Number of samples to evaluate.")
    parser.add_argument("--threshold", type=float, default=0.05, help="Threshold for COACD.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file.")
    args = parser.parse_args()

    config = load_config(args.config)

    if os.path.exists("stats.txt"):
        print("Stats file already exists. Please remove it before running the script.")
        exit()



    if args.vhacd:
        dataset = VHACD()
        for i in range(len(dataset)):
            p, st = dataset[i]
            mesh = coacd.Mesh(np.asarray(st.vertices), np.asarray(st.triangles))
            result = coacd.run_coacd(mesh,threshold=args.threshold)

    else:
        it = ACDgen(config,output_meshes=True).__iter__()
        set_seed(args.seed)
        #next(it)  

        for i in range(NUM_SAMPLES):
            points, distances_t, structure = next(it)

            mesh = coacd.Mesh(np.asarray(structure.vertices), np.asarray(structure.triangles))
            result = coacd.run_coacd(mesh)

    total_concavity = 0.0
    total_parts = 0

    with open("stats.txt", "r") as f:
        for line in f.readlines():
            concavity, num_parts = map(float, line.strip().split(';'))
            total_concavity += concavity
            total_parts += num_parts
    print(f"Average Concavity: {total_concavity / NUM_SAMPLES}")
    print(f"Average Parts: {total_parts / NUM_SAMPLES}")