import coacd_modified
import numpy as np
import trimesh
from utils.BaseUtils import *
import pyntcloud
import json
import datetime
import time

if __name__ == "__main__":

    c = 0

    with open("plane_cache.json", "r") as plane_cache_f:
        plane_cache = json.load(plane_cache_f)
        for key, value in plane_cache.items():
            for plane in value:
                rotation = trimesh.transformations.random_rotation_matrix()

                try:
                    rotated_plane = apply_rotation_to_plane(*plane[:4],rotation)
                except:
                    print(plane)
    print(c)
    
#     # plane_cache = json.load(open("plane_cache.json", "r"))

#     # for batch in load_shapenet(debug=True,batch_size=16):
#     #     for mesh in batch:
            
#     #         mesh_hash = str(hash((mesh.vertices, mesh.faces)))

#     #         if (mesh_hash in broken):
#     #             cmesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)

#     #             planes = coacd_modified.best_cutting_planes(cmesh, num_planes=5)

#     #             planes = [(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in planes]
#     #             print(str(hash((mesh.vertices, mesh.faces))),planes)
#     #             plane_cache[mesh_hash] = planes
    
#     # with open("plane_cache.json", "w") as plane_cache_f:
#     #     json.dump(plane_cache, plane_cache_f)
                

#     # mesh = trimesh.load("mesh.obj")
#     # cmesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)
#     # print(hash((mesh.vertices, mesh.faces)))
#     # planes = coacd_modified.best_cutting_planes(cmesh, num_planes=5)
#     # print([(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in planes])

#     # coacd_modified.run_coacd(cmesh)

#     p1 = json.load(open("1/plane_cache.json", "r"))
#     p2 = json.load(open("2/plane_cache.json", "r"))
#     p3 = json.load(open("3/plane_cache.json", "r"))
#     p4 = json.load(open("4/plane_cache.json", "r"))
#     p5 = json.load(open("5/plane_cache.json", "r"))

#     #combine
#     p1.update(p2)
#     p1.update(p3)
#     p1.update(p4)
#     p1.update(p5)
#     c = 0

#     for key, value in p1.items():
#         if len(value) < 5 or 'e' in str(value):
#             c+=1

    # with open("plane_cache.json", "r") as plane_cache_f:
    #     plane_cache = json.load(plane_cache_f)

    #     #iterate over files in tmp
    #     for root,dirs,files in os.walk("tmp2"):
    #         for file in files:
    #             if file.endswith(".obj"):
    #                 obj = trimesh.load(os.path.join(root,file))
    #                 #check if the obj contains a scene instead of a single mesh
    #                 if isinstance(obj,trimesh.Scene):
    #                     if len(obj.geometry) == 0:
    #                         continue
    #                     else:
    #                         mesh = trimesh.util.concatenate(
    #                             tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
    #                                 for g in obj.geometry.values()))
    #                 else:
    #                     mesh = obj
    #                 mesh_hash = str(hash((mesh.vertices, mesh.faces)))
    #                 cmesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)
    #                 res = coacd_modified.best_cutting_planes(cmesh, num_planes=5)
    #                 #compare if the plane in cache is the same as the one calculated
    #                 planes = [(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in res]
    #                 cache_planes = plane_cache[mesh_hash]
    #                 for i in range(5):
    #                     if planes[i][0] != cache_planes[i][0] or planes[i][1] != cache_planes[i][1] or planes[i][2] != cache_planes[i][2] or planes[i][3] != cache_planes[i][3]:
    #                         print("Mismatch")
    #                         print(planes[i])
    #                         print(cache_planes[i])
    #                         print(mesh_hash)
    #                         exit()
                    

    #     mesh = trimesh.load("tmp2/-1241765121852705049.obj")
    #     #mesh = trimesh.load("tmp/-11911506453957110.obj")
    #     mesh = trimesh.util.concatenate(
    #                                 tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
    #                                     for g in mesh.geometry.values()))
    #     print(plane_cache[str(hash((mesh.vertices, mesh.faces)))])

    #     cmesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)

    #     res = coacd_modified.normalize(cmesh)

    #     mesh = trimesh.Trimesh(res.vertices, res.indices)

    #     mesh.export("normalized.obj")

    #     res = coacd_modified.best_cutting_planes(cmesh, num_planes=5)

    #     planes = [(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in res]
    #     print(planes)


        #res = coacd_modified.run_coacd(cmesh)

        # mesh_parts = []

        # for vs, fs in res:
        #     mesh_parts.append(trimesh.Trimesh(vs, fs))

        # scene = trimesh.Scene()
        # np.random.seed(0)
        # for p in mesh_parts:
        #     p.visual.vertex_colors[:, :3] = (np.random.rand(3) * 255).astype(np.uint8)
        #     scene.add_geometry(p)
        # scene.export("decomposed.obj")

