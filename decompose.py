import torch
import open3d as o3d
import coacd_modified
import numpy as np
from model import model
import random
from concurrent.futures import ThreadPoolExecutor,ProcessPoolExecutor
import datetime
import os
import sys
import trimesh
import copy
import shutil


MODEL_PATH =  "model/best_model.pth"
#np.random.seed(0)



def revert_normalization(mesh,bbox):
    x_min, x_max, y_min, y_max, z_min, z_max = bbox

    m_len = max(x_max - x_min, y_max - y_min, z_max - z_min)
    m_Xmid = (x_max + x_min) / 2
    m_Ymid = (y_max + y_min) / 2
    m_Zmid = (z_max + z_min) / 2

    vertices = np.asarray(mesh.vertices)

    # Scale and translate vertices
    vertices = vertices / 2 * m_len + np.array([m_Xmid, m_Ymid, m_Zmid])

    # Update mesh
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.compute_vertex_normals() 


def normalize_meshes(meshes,make_manifold=True):
    bboxes = []
    normalized_meshes = []
    normalized_cmeshes = []
    for mesh in meshes:
        cmesh = coacd_modified.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
        bbox,normalized_cmesh = coacd_modified.normalize(cmesh,make_manifold=make_manifold)
        bboxes.append(bbox)
        normalized_mesh = o3d.geometry.TriangleMesh()
        normalized_mesh.vertices = o3d.utility.Vector3dVector(normalized_cmesh.vertices)
        normalized_mesh.triangles = o3d.utility.Vector3iVector(normalized_cmesh.indices)
        normalized_meshes.append(normalized_mesh)
        normalized_cmeshes.append(normalized_cmesh)

    return bboxes,normalized_meshes,normalized_cmeshes

def get_point_clouds(meshes):
    point_clouds = []
    for mesh in meshes:
        pnt = mesh.sample_points_poisson_disk(number_of_points=512, init_factor=3)
        points = np.asarray(pnt.points)
        point_clouds.append(points)
    
    return np.array(point_clouds)

def normalize_planes(planes):
    new_planes = []
    for i in range(len(planes)):
        plane = planes[i]
        plane_abc = plane[:3]
        plane_d = plane[3]

        plane_abc = plane_abc / torch.norm(plane_abc)
        plane_abc[0] = 0
        plane_abc[1] = 0
        plane_abc[2] = 1
        plane = torch.cat([plane_abc, plane_d.view(1)])
        plane = coacd_modified.CoACD_Plane(*list(plane.cpu().numpy()),0)
        new_planes.append(plane)
    
    return new_planes


def cut_mesh(cmesh,plane,bbox):
    # mesh = o3d.geometry.TriangleMesh()
    # mesh.vertices = o3d.utility.Vector3dVector(cmesh.vertices)
    # mesh.triangles = o3d.utility.Vector3iVector(cmesh.indices)
    # mesh.compute_vertex_normals()
    # o3d.io.write_triangle_mesh("temp_mesh.ply", mesh)
    # print(plane.a,plane.b,plane.c,plane.d)
    result = coacd_modified.clip(cmesh, plane)
    parts = []
    for vs,fs in result:
        if len(vs) == 0 or len(fs) == 0:
            continue
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vs)
        mesh.triangles = o3d.utility.Vector3iVector(fs)
        revert_normalization(mesh,bbox)





        #if the produced mesh contains disconnected parts, we need to split them

        parts.append(mesh)
        
        # triangle_clusters, cluster_n_triangles, cluster_area = mesh.cluster_connected_triangles()

        

        # if (len(cluster_n_triangles) == 1):
        #     parts.append(mesh)
        # else:
        #     triangle_clusters = np.asarray(triangle_clusters)
        #     for i in range(len(cluster_n_triangles)):
        #         triangles_to_remove = triangle_clusters != i
        #         new_mesh = copy.deepcopy(mesh)
        #         new_mesh.remove_triangles_by_mask(triangles_to_remove)
        #         new_mesh.remove_unreferenced_vertices()

        #         parts.append(new_mesh)

    

    return parts

def get_convex_hulls(meshes):
    hulls = []

    concavity_sum = 0

    for part in meshes:
        cmesh = coacd_modified.Mesh(np.asarray(part.vertices), np.asarray(part.triangles))
        bbox,normalized_cmesh = coacd_modified.normalize(cmesh)
        hull, concavity = coacd_modified.compute_convex_hull(normalized_cmesh)

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(hull.vertices)
        mesh.triangles = o3d.utility.Vector3iVector(hull.indices)
        revert_normalization(mesh,bbox)
        concavity_sum += concavity
        hulls.append(mesh)

    return hulls,concavity_sum/len(meshes)



def decompose(mesh,depth,predictor,min_part_size=40): 
    #o3d.io.write_triangle_mesh("input_mesh.ply", mesh)   

    parts = [mesh]        



    for _ in range(depth):
        new_parts = []
        
        bboxes,normalized_meshes,normalized_cmeshes = normalize_meshes(parts)

        # os.makedirs("tmp", exist_ok=True)
        # for mesh in normalized_meshes:
        #     o3d.io.write_triangle_mesh(f"tmp/{datetime.datetime.now().timestamp()}.ply", mesh)

        clouds = get_point_clouds(normalized_meshes)

        clouds = clouds.reshape(-1, 3, 512)

        clouds = torch.tensor(clouds, dtype=torch.float32).cuda()

        with torch.no_grad():
            planes = predictor(clouds)

            planes = normalize_planes(planes)

        executor = ThreadPoolExecutor()
        futures = []
        for i in range(len(parts)):
            if len(normalized_meshes[i].triangles) < min_part_size:
                new_parts.append(parts[i])
                continue
            futures.append(executor.submit(cut_mesh,normalized_cmeshes[i],planes[i],bboxes[i]))
        
        for future in futures:
            new_parts += future.result()


        parts = new_parts
        if len(parts) > 2**depth:
            break

    hulls,avg_concavity = get_convex_hulls(parts)

    #shutil.rmtree("tmp", ignore_errors=True)

    return parts,hulls,avg_concavity
    

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Please provide a mesh file")
        exit(0)

    coacd_modified.set_log_level("debug")
    
    filename = sys.argv[1]

    if len(sys.argv) > 2:
        depth = int(sys.argv[2])
    else:
        depth = 5
        print("Using default depth of 5")

    predictor = model.get_model(4).cuda()

    predictor.load_state_dict(torch.load(MODEL_PATH)['model_state_dict'])

    mesh = o3d.io.read_triangle_mesh(filename)
    parts,hulls,avg_concavity = decompose(mesh,depth,predictor)
    print("avg concavity: ",avg_concavity)
    print("number of parts: ",len(parts))

    scene = trimesh.Scene()

    for hull in hulls:
        scene.add_geometry(trimesh.Trimesh(vertices=np.asarray(hull.vertices), faces=np.asarray(hull.triangles)))
    
    scene.export("decomposed.obj")
    print("Decomposed mesh saved to: decomposed.obj")



