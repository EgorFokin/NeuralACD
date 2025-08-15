import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..","lib", "build"))

from scripts.mark_cuts import mark_cuts, threshold_values
import lib_neural_acd
from utils.ACDgen import ACDgen
import trimesh
import numpy as np
import argparse
from utils.misc import *
import coacd
import tempfile
import shutil
import json
from utils.visualization import *
from scripts.decompose import decompose
from utils.VHACD import VHACD

coacd.set_log_level("off")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a dataset using ACD models and visualize results.")
    parser.add_argument("--path", type=str, default="", help="Path to the dataset.")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoint.ckpt", help="Path to the model checkpoint.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to the configuration file.")
    parser.add_argument("--coacd", action="store_true", help="Add CoACD results")
    parser.add_argument("--vhacd", type=int, default=0, help="Add VHACD dataset")
    parser.add_argument("--cuboids", type=int, default=0, help="Number of cuboids to generate")
    parser.add_argument("--spheres", type=int, default=0, help="Number of spheres to generate")
    parser.add_argument("--out", type=str, default="out/visual", help="Output directory for results")
    
    args = parser.parse_args()

    config = load_config(args.config)

    
    shutil.rmtree(args.out, ignore_errors=True)
    os.makedirs(args.out, exist_ok=True)
    

    data = []

    if args.spheres + args.cuboids > 0:
        print("Generating synthetic meshes...")

        acdgen = ACDgen(config, output_meshes=True)
        it = acdgen.__iter__()
        
        acdgen.config.generation.struct_types =["sphere"]
        for i in range(args.spheres):
            seed = args.seed + i
            set_seed(seed)
            points, distances_t, structure = next(it)

            coacd_mesh = coacd.Mesh(np.asarray(structure.vertices), np.asarray(structure.triangles))

            data.append({"lib_mesh": structure, "pcd": points, "coacd_mesh":coacd_mesh, "name": f"sphere_{seed}"})

        acdgen.config.generation.struct_types =["cuboid"]
        for i in range(args.cuboids):
            seed = args.seed + i
            set_seed(seed)
            points, distances_t, structure = next(it)

            coacd_mesh = coacd.Mesh(np.asarray(structure.vertices), np.asarray(structure.triangles))

            data.append({"lib_mesh": structure, "pcd": points, "coacd_mesh":coacd_mesh, "name": f"cuboid_{seed}"})

    
    vhacd = VHACD(config)
    for i in range(args.vhacd):
        points, structure, name = vhacd[i]
        coacd_mesh = coacd.Mesh(np.asarray(structure.vertices), np.asarray(structure.triangles))
        data.append({"lib_mesh": structure, "pcd": points, "coacd_mesh":coacd_mesh, "name": name})


    for mesh_data in data:
        os.makedirs(os.path.join(args.out,mesh_data["name"]), exist_ok=True)
        

        tmesh = trimesh.Trimesh(vertices=np.asarray(mesh_data["lib_mesh"].vertices), faces=np.asarray(mesh_data["lib_mesh"].triangles))

        tmesh.export(os.path.join(args.out,mesh_data["name"], "original.glb"))

        scene = trimesh.Scene()
        scene.add_geometry(tmesh)
        render_scene(scene, os.path.join(args.out, mesh_data["name"], "render.png"))


    results = {}
    for mesh_data in data:
        results[mesh_data["name"]] = {}

        os.makedirs(os.path.join(args.out,mesh_data["name"], "neural_acd"), exist_ok=True)

        structure = mesh_data["lib_mesh"]
        points = mesh_data["pcd"]
        values = mark_cuts(points, args.checkpoint, config,no_threshold=True)

        render_pcd(points[:,:3], values, os.path.join(args.out, mesh_data["name"],"neural_acd", "prediction.png"))
        values = threshold_values(np.expand_dims(values,axis=0), config)
        values = values.squeeze(0)
        

        with tempfile.NamedTemporaryFile(mode='r+') as temp_file:
            parts = decompose(structure, points[values == 1], temp_file.name)
            concavity, num_parts = map(float, temp_file.readline().strip().split(';'))

        scene = trimesh.Scene()
        for part in parts:
            triangles = np.asarray(part.triangles)
            vertices = np.asarray(part.vertices)
            if (triangles.shape[0] == 0 or vertices.shape[0] == 0):
                continue
            tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles)
            tmesh.visual.face_colors = np.random.randint(0, 255, (3), dtype=np.uint8)
            scene.add_geometry(tmesh)

        results[mesh_data["name"]]["neural_acd"]= {"concavity": concavity, "num_parts": num_parts}

        
        scene.export(os.path.join(args.out,mesh_data["name"], "neural_acd", "decomposed.glb"))
        render_scene(scene, os.path.join(args.out,mesh_data["name"], "neural_acd", "render.png"))


    

    if args.coacd:
        print("Decomposing using CoACD...")
        for mesh_data in data:
            coacd_mesh = mesh_data["coacd_mesh"]
            with tempfile.NamedTemporaryFile(mode='r+') as temp_file:
                stats_file = temp_file.name
                result = coacd.run_coacd(coacd_mesh, threshold=0.05,stats_file=stats_file)

                concavity, num_parts = map(float, temp_file.readline().strip().split(';'))


            scene = trimesh.Scene()
            for vs, fs in result:
                tmesh = trimesh.Trimesh(vertices=np.asarray(vs), faces=np.asarray(fs))
                tmesh.visual.face_colors = np.random.randint(0, 255, (3), dtype=np.uint8)
                scene.add_geometry(tmesh)

            os.makedirs(os.path.join(args.out,mesh_data["name"], "coacd"), exist_ok=True)
    
            scene.export(os.path.join(args.out,mesh_data["name"], "coacd", "decomposition.glb"))
            render_scene(scene, os.path.join(args.out,mesh_data["name"], "coacd", "render.png"))


            results[mesh_data["name"]]["coacd"]= {"concavity": concavity, "num_parts": num_parts}
    
    

    json.dump(results, open(os.path.join(args.out, "results.json"), "w"), indent=4)
