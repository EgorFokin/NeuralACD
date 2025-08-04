import torch
import open3d as o3d
from torch.utils.data import IterableDataset
import numpy as np
import lib_neural_acd
import random
import trimesh
from scipy.spatial import KDTree, cKDTree


class ACDgen(IterableDataset):
    def __init__(self, config, output_meshes=False):
        super().__init__()
        self.config = config
        self.output_meshes = output_meshes
    

    def get_distances(self,pcd_points, cut_points):
        if len(cut_points) == 0:
            return np.ones(len(pcd_points), dtype=np.float32)
        
        tree = cKDTree(cut_points)
        dists, _ = tree.query(pcd_points, k=1)  

        return dists.astype(np.float32)
    
    def apply_gaussian_filter(self, pcd, structure, sigma=0.02, radius = 0.05):
        points = np.vstack([pcd,np.asarray(structure.vertices)])

        kdtree = KDTree(points)

        filtered_points = []
        for point in points:
            indices = kdtree.query_ball_point(point, radius)
            if len(indices) > 0:
                neighbors = points[indices]
                weights = np.exp(-np.linalg.norm(neighbors - point, axis=1) ** 2 / (2 * sigma ** 2))
                filtered_point = np.sum(weights[:, np.newaxis] * neighbors, axis=0) / np.sum(weights)
                filtered_points.append(filtered_point)
            else:
                filtered_points.append(point)
        
        structure.vertices = lib_neural_acd.VecArray3d(filtered_points[-len(structure.vertices):])
        return np.asarray(filtered_points[:len(pcd)])

    def apply_random_scale(self, points, structure):
        scale_x = np.random.uniform(1, 3.0)
        scale_y = np.random.uniform(1, 3.0)
        scale_z = np.random.uniform(1, 3.0)

        stretch_matrix = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, scale_z]
        ])

        points = points @ stretch_matrix.T


        structure.vertices = lib_neural_acd.VecArray3d(np.asarray(structure.vertices) @ stretch_matrix.T)
    
        return points

    def apply_random_rotation(self, points, structure):
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        points = points @ rotation_matrix.T

        structure.vertices = lib_neural_acd.VecArray3d(np.asarray(structure.vertices) @ rotation_matrix.T)
        return points

    def normalize_mesh(self, points, structure):
        min_vals = points.min(axis=0)
        centered = points - min_vals
        scale = np.max(centered.max(axis=0))  # max range across all axes
        points_normalized = centered / scale

        struct_vertices = (np.asarray(structure.vertices) - min_vals) / scale
        structure.vertices = lib_neural_acd.VecArray3d(struct_vertices)

        return points_normalized
            
    def __iter__(self):
        num_spheres = np.random.randint(self.config.generation.min_parts, self.config.generation.max_parts + 1)
        while True:
            structure_type = np.random.choice(['sphere', 'cuboid'])
            if structure_type == 'sphere':
                structure = lib_neural_acd.generate_sphere_structure(num_spheres)
            elif structure_type == 'cuboid':
                structure = lib_neural_acd.generate_cuboid_structure(num_spheres)


            
            #print(verts)
            # if self.output_meshes:
            lib_neural_acd.preprocess(structure, 100.0, 0.001)
                


            points = lib_neural_acd.VecArray3d()
            point_tris = lib_neural_acd.VecInt()
            structure.extract_point_set(points, point_tris, self.config.general.num_points)

            points = np.asarray(points)

            # print(points.shape, len(structure.triangles), len(structure.vertices))

            distances = self.get_distances(points, structure.cut_verts)

            if self.config.generation.random_scale and structure_type == 'sphere':
                points = self.apply_random_scale(points, structure)
            if self.config.generation.random_rotation:
                points = self.apply_random_rotation(points,structure)
            if self.config.generation.gaussian_filter:
                radius = np.random.uniform(0.0, 0.05) if structure_type == 'cuboid' else np.random.uniform(0.0, 0.1)
                points = self.apply_gaussian_filter(points, structure,sigma=0.1,radius=radius)
            
            
            if self.config.generation.normalize_mesh:
                points = self.normalize_mesh(points,structure)

            

            mesh = trimesh.Trimesh(vertices=structure.vertices, faces=structure.triangles)

            curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, points, 0.02)

            normals = mesh.face_normals[np.asarray(point_tris)]

            # pcd = trimesh.PointCloud(points)
            # #color normals
            # colors = np.zeros((len(points), 3))
            # colors[:, 0] = (normals[:, 0] + 1) / 2
            # colors[:, 1] = (normals[:, 1] + 1) / 2
            # colors[:, 2] = (normals[:, 2] + 1) / 2
            # pcd.colors = o3d.utility.Vector3dVector(colors)
            # pcd.show()

            #add curvature channel
            points = np.hstack((points, curvature[:, np.newaxis], normals))



            points = torch.tensor(points, dtype=torch.float32)
            distances = torch.tensor(distances, dtype=torch.float32)
            distances = 1 - (torch.clamp(distances, 0.01, 0.03)-0.01)*50
            # distances[distances != 0] = 1

            if self.output_meshes:
                yield points, distances, structure
            else:
                yield points, distances