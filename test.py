import coacd_modified
import coacd
import numpy as np
import trimesh
from utils.BaseUtils import *
import pyntcloud

if __name__ == "__main__":

    for mesh in load_shapenet(debug=True):
        mesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)
        result = coacd_modified.normalize(mesh)

        plane = coacd_modified.best_cutting_plane(mesh,merge=False)
        print(plane.a, plane.b, plane.c, plane.d)

        normalized_mesh = trimesh.Trimesh(result.vertices, result.indices)

        normalized_mesh.export("normalized.ply")


        rotation = apply_random_rotation(normalized_mesh)
        print(rotation)
        plane = apply_rotation_to_plane(plane.a, plane.b, plane.c, plane.d,rotation)

        print(plane)

        normalized_mesh.export("rotated.ply")
        

        pc_mesh = pyntcloud.PyntCloud.from_file("rotated.ply")

        pc = pc_mesh.get_sample("mesh_random", n=512, as_PyntCloud=True)

        points = np.array(pc.points)

        print(points)


    # mesh = trimesh.load("rotated_thing.obj")
    # mesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)

    # normilized_mesh = coacd_modified.normalize(mesh)
    # scene = trimesh.Scene()
    # scene.add_geometry(trimesh.Trimesh(normilized_mesh.vertices, normilized_mesh.indices))
    # scene.export("normalized.obj")

    # result = coacd_modified.run_coacd(mesh)
    # mesh_parts = []
    # for vs, fs in result:
    #     mesh_parts.append(trimesh.Trimesh(vs, fs))

    # scene = trimesh.Scene()
    # np.random.seed(0)
    # for p in mesh_parts:
    #     p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
    #     scene.add_geometry(p)
    # scene.export("decomposed.obj")
    

    # plane = coacd_modified.best_cutting_plane(mesh,merge=False)
    # print(plane.a, plane.b, plane.c, plane.d)

    # result = coacd_modified.clip(normilized_mesh,plane)
    # mesh_parts = []
    # for vs, fs in result:
    #     mesh_parts.append(trimesh.Trimesh(vs, fs))

    # scene = trimesh.Scene()
    # np.random.seed(0)
    # for p in mesh_parts:
    #     p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
    #     scene.add_geometry(p)
    # scene.export("decomposed.obj")