import trimesh
import argparse
import os
import json
import shutil


def is_colored(mesh_path):
    obj = trimesh.load(mesh_path)
    mesh = list(obj.geometry.values())[0]
    return mesh.visual.kind == 'vertex'


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Filter colliders from a mesh.")
    parser.add_argument("--path", type=str, required=True, help="Path to the mesh file.")
    parser.add_argument("--output", type=str, required=True, help="Path to save the filtered mesh.")
    
    args = parser.parse_args()

    for partition in ["0","1","2","3","4","5","6","7","8","9"]:
        for obj in os.listdir(os.path.join(args.path, "objects",partition)):
            if obj.endswith(".json"):
                with open(os.path.join(args.path, "objects", partition, obj), 'r') as f:
                    data = json.load(f)
                if "collision_asset" not in data or "render_asset" not in data:
                    continue
                collider_path = data["collision_asset"]
                mesh_path = data["render_asset"]

                if is_colored(os.path.join(args.path, "objects", partition, collider_path)):
                    continue

                id = mesh_path.split('.')[0]
                os.makedirs(os.path.join(args.output,id), exist_ok=True)
                shutil.copy(os.path.join(args.path, "objects", partition, collider_path), os.path.join(args.output, id, "collider.glb"))
                shutil.copy(os.path.join(args.path, "objects", partition, mesh_path), os.path.join(args.output, id, "mesh.glb"))
