import coacd_modified
import numpy as np
import trimesh
from utils.BaseUtils import *
import pyntcloud
import json
import datetime
import time

# if __name__ == "__main__":

#     # with open("plane_cache.json", "r") as plane_cache_f:
#     #     plane_cache = json.load(plane_cache_f)
#     #     for key, value in plane_cache.items():
#     #         for plane in value:
#                 # rotation = trimesh.transformations.random_rotation_matrix()

#                 # try:
#                 #     rotated_plane = apply_rotation_to_plane(*plane[:4],rotation)
#                 # except:
#                 #     print(plane)
#                 #     exit()
#                 # print(rotated_plane[0]/rotated_plane[3],rotated_plane[1]/rotated_plane[3],rotated_plane[2]/rotated_plane[3])
    
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

#     mesh = trimesh.load("tmp/-11911506453957110.obj")
#     mesh = trimesh.util.concatenate(
#                                 tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
#                                     for g in mesh.geometry.values()))
#     print(p1[str(hash((mesh.vertices, mesh.faces)))])

#     cmesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)
#     res = coacd_modified.best_cutting_planes(cmesh, num_planes=5)
#     print([(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in res])

#     res = coacd_modified.normalize(cmesh)

#     mesh = trimesh.Trimesh(res.vertices, res.indices)

#     mesh.export("normalized.obj")


TOTAL_MESHES = 52000

if __name__ == "__main__":
    i = 0
    batch_size = 16
    start_time = datetime.datetime.now()
    time_spent = datetime.timedelta()
    ten_prev = []
    for i in range(100000):
        i+=1
        time.sleep(1)
        delta = datetime.datetime.now() - start_time


        ten_prev.append(delta)
        time_spent += delta
        if i>10:
            print(f"mesh{i*batch_size}/{TOTAL_MESHES}; {str(time_spent).split('.')[0]}/{str(sum(ten_prev, datetime.timedelta())/10*(TOTAL_MESHES/(batch_size))).split('.')[0]}")
            ten_prev.pop(0)
        else:
            print(f"mesh{i*batch_size}/{TOTAL_MESHES}; {str(time_spent).split('.')[0]}/{str(delta*(TOTAL_MESHES/(batch_size))).split('.')[0]}")

        start_time = datetime.datetime.now()