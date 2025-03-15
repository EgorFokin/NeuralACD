import coacd_modified
import numpy as np
import trimesh
from utils.BaseUtils import *
#import pyntcloud
import json
import datetime
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from model import model
from torchviz import make_dot
import open3d as o3d


# def forward(pred, target):


#         #target_norm[..., 3] = target_norm[..., 3] * 2 - 1 # d is always positive, so we need to convert it to the range [-1, 1]


#         #distance to the closest plane
#         loss_fn = nn.MSELoss(reduction='none')  # Compute element-wise MSE


#         # Compute MSE loss for all target planes
#         loss = loss_fn(pred.unsqueeze(1), target)  # Shape: (B, M, 4)

#         # Sum over the last dimension (MSE is applied to 4D vectors)
#         mse = loss.mean(dim=-1)  # Shape: (B, M)


#         #compare normal directions
#         pred_dir = pred[..., :3]  # Shape: (B, 3)
#         target_dir = target[..., :3]  # Shape: (B, M, 3)

#         # Normalize the direction vectors to compare only their directions
#         pred_dir_norm = F.normalize(pred_dir, dim=-1)  # Shape: (B, 3)
#         target_dir_norm = F.normalize(target_dir, dim=-1)  # Shape: (B, M, 3)

#         # Compute cosine similarity (higher means better alignment)
#         cosine_sim = torch.matmul(pred_dir_norm.unsqueeze(1), target_dir_norm.transpose(-1, -2)).squeeze(1)  # Shape: (B, M)

#         # Convert similarity to loss (1 - similarity, so lower is better)
#         cosine = 1 - cosine_sim  # Shape: (B, M)

        

#         loss = mse + 0.5*cosine

#         # Take the minimum loss over the M target planes
#         min_loss, _ = loss.min(dim=1)  # Shape: (B,)
#         normalization_loss = torch.abs((1 - torch.norm(pred_dir, dim=-1)))

#         print("mse:",mse)
#         print("cosine:",cosine)
#         print("loss:",loss)
#         print("min_loss:",min_loss)
#         print("normalization_loss:",normalization_loss)
#         return min_loss.mean() + normalization_loss.mean()


# os.environ["PYTHONHASHSEED"] = "0"

# os.environ["PATH"] += os.pathsep + 'C:\\Program Files (x86)\\Graphviz\\bin'

# MODEL_PATH = "C:\\Users\\egorf\\Desktop\\cmpt469\\DeepConvexDecomposition\\log\\2025-03-13_15-15\\checkpoints\\checkpoint.pth"

# predictor = model.get_model(4).cuda()

# predictor.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])

# from prettytable import PrettyTable

# with open("data/ShapeNetPointCloud_val/ffbf17fbe4ca367d54cd2a0ea6cb618b.npy","rb") as f:
#     points = np.load(f)

# with open("plane_cache.json", "r") as plane_cache_f:
#     cache = json.load(plane_cache_f)

# pnt = o3d.geometry.PointCloud()
# pnt.points = o3d.utility.Vector3dVector(points)

# planes = cache["ffbf17fbe4ca367d54cd2a0ea6cb618b"]

# rotation = np.eye(3)

# planes = [apply_rotation_to_plane(*plane[:4],rotation) for plane in planes]

# #print(torch.tensor(planes))
# points = points.reshape(1, -1, 512)

# predicted = predictor(torch.tensor(points,dtype=torch.float32).cuda())

# # print(predicted)

# # print(predicted.norm(dim=-1))

# target = torch.tensor(planes,dtype=torch.float32).cuda()

# target = target.reshape(1, -1, 4)

# print(predicted)
# print(target)
# print(forward(predicted,target))




#o3d.visualization.draw_geometries([pnt])

# def forward(pred, target):

#     target_norm = F.normalize(target, dim=-1)
#     #target_norm[..., 3] = target_norm[..., 3] * 2 - 1 # d is always positive, so we need to convert it to the range [-1, 1]


#     #distance to the closest plane
#     loss_fn = nn.MSELoss(reduction='none')  # Compute element-wise MSE
#     print(pred.unsqueeze(1))
#     print(target_norm)
#     print("-------------------")

#     # Compute MSE loss for all target planes
#     loss = loss_fn(pred.unsqueeze(1), target_norm)  # Shape: (B, M, 4)

#     # Sum over the last dimension (MSE is applied to 4D vectors)
#     mse = loss.mean(dim=-1)  # Shape: (B, M)


#     #compare normal directions
#     pred_dir = pred[..., :3]  # Shape: (B, 3)
#     target_dir = target[..., :3]  # Shape: (B, M, 3)

#     # Normalize the direction vectors to compare only their directions
#     pred_dir_norm = F.normalize(pred_dir, dim=-1)  # Shape: (B, 3)
#     target_dir_norm = F.normalize(target_dir, dim=-1)  # Shape: (B, M, 3)

#     # Compute cosine similarity (higher means better alignment)
#     cosine_sim = torch.matmul(pred_dir_norm.unsqueeze(1), target_dir_norm.transpose(-1, -2)).squeeze(1)  # Shape: (B, M)

#     # Convert similarity to loss (1 - similarity, so lower is better)
#     cosine = 1 - cosine_sim  # Shape: (B, M)

#     loss = mse + cosine

#     # Take the minimum loss over the M target planes
#     min_loss, _ = loss.min(dim=1)  # Shape: (B,)

#     print("mse:",mse)
#     print("cosine:",cosine)
#     print("min_loss:",min_loss)
#     print("-------------------")


#     return min_loss.mean()

# # Similar planes (should minimize loss)
# similar_planes_pred = F.normalize(torch.tensor([
#     [0.707, 0.707, 0, 5],   # Identical planes
#     [0.705, 0.709, 0, 5.1], # Slightly perturbed normal
#     [1, 0, 0, 3.2]          # Parallel planes with slight offset
# ]),dim=-1)

# similar_planes_target = torch.tensor([
#     [[0.707, 0.707, 0, 5], [1, 0, 0, 3.2]],   # Identical
#     [[0.707, 0.707, 0, 5],[0.705, 0.709, 0, 5.1]],   # Slightly perturbed normal
#     [[1, 0, 0, 3], [1, 0, 0, 3.2] ]            # Parallel
# ])

# # Dissimilar planes (should maximize loss)
# dissimilar_planes_pred =  F.normalize(torch.tensor([
#     [0, -1, 0, 2],  # Opposite normals
#     [0, 1, 0, 1],   # Perpendicular normals
#     [0, 0, 1, 100], # Extreme offset difference
#     [-0.3, 0.1, 0.95, 20]  # Completely random normal
# ]),dim=-1)

# dissimilar_planes_target = torch.tensor([
#     [[0, 1, 0, 2]],  # Opposite normals
#     [[1, 0, 0, 1]],  # Perpendicular normals
#     [[0, 0, 1, 2]],  # Extreme offset difference
#     [[0.6, 0.8, 0, 4]]  # Completely random normal
# ])

# # Test similar planes
# loss = forward(similar_planes_pred, similar_planes_target)
# print(f"Similar planes loss: {loss.item()}")

# # Test dissimilar planes
# loss = forward(dissimilar_planes_pred, dissimilar_planes_target)
# print(f"Dissimilar planes loss: {loss.item()}")




# m = model.get_model(4).cuda()

# x = torch.randn(16,3,512).cuda()
# print(x.shape)
# y = m(x)



# make_dot(y.mean(), params=dict(m.named_parameters())).render("viz", format="png")


# def normalize_mesh(mesh):
#     # Get the vertex positions as a NumPy array
#     vertices = np.asarray(mesh.vertices)

#     # Compute bounding box
#     x_min, y_min, z_min = vertices.min(axis=0)
#     x_max, y_max, z_max = vertices.max(axis=0)

#     # Compute max length and midpoints
#     m_len = max(x_max - x_min, y_max - y_min, z_max - z_min)
#     m_Xmid = (x_max + x_min) / 2
#     m_Ymid = (y_max + y_min) / 2
#     m_Zmid = (z_max + z_min) / 2

#     # Normalize vertices
#     vertices = 2.0 * (vertices - np.array([m_Xmid, m_Ymid, m_Zmid])) / m_len
#     mesh.vertices = o3d.utility.Vector3dVector(vertices)

#     return mesh

# if __name__ == "__main__":
#     # mesh = o3d.io.read_triangle_mesh("broken_mesh.obj")
#     # cmesh = coacd_modified.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
#     # result = coacd_modified.normalize(cmesh)
#     # result2 = normalize_mesh(mesh)

#     # #check that the vertices are the same
#     # print(np.allclose(np.asarray(result.vertices),np.asarray(mesh.vertices)))

#     # #check that the faces are the same
#     # print(np.allclose(np.asarray(result.indices),np.asarray(mesh.triangles)))
#     with open("data/ShapeNetPointCloud/1a00aa6b75362cc5b324368d54a7416f.npy", "rb") as f:
#         points = np.load(f)
#         rotation = o3d.geometry.get_rotation_matrix_from_xyz(np.random.rand(3) * 2 * np.pi)
#         print(rotation)
#         print(points[0])
#         points = np.dot(points, rotation[:3,:3].T)
#         print(points[0])





# if __name__ == "__main__":

#     obj = trimesh.load("ex.obj")
#     #check if the obj contains a scene instead of a single mesh
#     if isinstance(obj,trimesh.Scene):

#         mesh = trimesh.util.concatenate(
#             tuple(trimesh.Trimesh(vertices=g.vertices, faces=g.faces)
#                 for g in obj.geometry.values()))
#     else:
#         mesh = obj

#     cmesh = coacd_modified.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
#     result = coacd_modified.normalize(cmesh)

#     planes = coacd_modified.best_cutting_planes(cmesh, num_planes=5)
#     print([(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in planes])

#     print()

#     m = coacd_modified.run_coacd(cmesh)
#     trimesh.Trimesh(m[0][0],m[0][1]).export("decomposed.obj")

#     trimesh.Trimesh(result.vertices, result.indices).export("normalized.obj")


# def count_parameters(m):
#     table = PrettyTable(["Modules", "Parameters"])
#     total_params = 0
#     for name, parameter in m.named_parameters():
#         if not parameter.requires_grad:
#             continue
#         params = parameter.numel()
#         table.add_row([name, params])
#         total_params += params
#     print(table)
#     print(f"Total Trainable Params: {total_params}")
#     return total_params


# if __name__ == "__main__":

#     p = model.get_model(4).cuda()
#     count_parameters(p)

    # with open("plane_cache.json", "r") as plane_cache_f:
    #     cache1 = json.load(plane_cache_f)

    # cache2 = dict()


    # for mesh_hash, mesh in load_shapenet(debug=False,data_folder="data/ShapeNetRedistributed"):
    #     hash2 = str(hash((mesh.vertices, mesh.faces)))
    #     cache2[mesh_hash] = cache1[hash2]

    # with open("plane_cache2.json", "w") as plane_cache_f:
    #     json.dump(cache2,plane_cache_f)


# if __name__ == "__main__":

#     c = 0
#     sm = 0

#     with open("plane_cache.json", "r") as plane_cache_f:
#         plane_cache = json.load(plane_cache_f)
#         for key,value in plane_cache.items():
#             if len(value) < 5:
#                 continue
#             planes = [p[:4] for p in value]
#             for i in range(len(planes)):
#                 if planes[i][3]<0:
#                     planes[i] = [-p for p in planes[i]]
#             plane = torch.tensor(planes,dtype=torch.float32)
#             plane = F.normalize(plane, dim=-1)
#             random_plane = [random.uniform(-1,1),random.uniform(-1,1),random.uniform(-1,1),random.uniform(0,1)]

#             random_plane = torch.tensor(random_plane,dtype=torch.float32)
#             random_plane = F.normalize(random_plane, dim=-1)
#             #print(plane.shape,random_plane.shape)
#             loss_fn = nn.MSELoss(reduction='none')  # Compute element-wise MSE

#             # Compute MSE loss for all target planes
#             loss = loss_fn(random_plane.unsqueeze(0), plane)  # Shape: (B, M, 4)

#             # Sum over the last dimension (MSE is applied to 4D vectors)
#             loss = loss.mean(dim=-1)  # Shape: (B, M)

#             # Take the minimum loss over the M target planes for each batch
#             min_loss, _ = loss.min(dim=-1) 
#             sm+=min_loss.mean().item()
#             c+=1
#         print(sm/c)
            
                
    #     for key, value in plane_cache.items():
    #         for plane in value:
    #             rotation = trimesh.transformations.random_rotation_matrix()

    #             try:
    #                 rotated_plane = apply_rotation_to_plane(*plane[:4],rotation)
    #                 a = rotated_plane[0]/rotated_plane[3]
    #                 b = rotated_plane[1]/rotated_plane[3]
    #                 c = rotated_plane[2]/rotated_plane[3]

    #                 mn = min(mn, min(a,b,c))
    #                 mx = max(mx, max(a,b,c))
    #                 #print(rotated_plane)
    #             except:
    #                 print(plane)
    #                 del plane_cache[key]
    #                 break
    # print(mx,mn)

    # with open("plane_cache.json", "w") as plane_cache_f:
    #     json.dump(plane_cache, plane_cache_f)

    # loss_fn = nn.MSELoss()
    # i1 = torch.tensor([1,1,1],dtype=torch.float32)
    # i2 = torch.tensor([0,0,0],dtype=torch.float32)
    # loss= loss_fn(i1,i2)
    # print(loss.item())
    
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

