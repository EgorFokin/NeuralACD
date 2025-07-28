import torch
import open3d as o3d
from torch.utils.data import IterableDataset
import numpy as np
import lib_neural_acd
import random
import trimesh

from utils.misc import get_point_cloud

class ACDgen(IterableDataset):
    def __init__(self, config, output_meshes=False):
        super().__init__()
        self.config = config
        self.output_meshes = output_meshes
    

    def get_distances(self, pcd_points,cut_points):
        if len(cut_points) == 0:
            return np.ones(len(pcd_points), dtype=np.float32)
        # Create a KD-tree from x
        cut_pcd = o3d.geometry.PointCloud()
        cut_pcd.points = o3d.utility.Vector3dVector(cut_points)
        kdtree = o3d.geometry.KDTreeFlann(cut_pcd)

        distances = []

        for p in pcd_points:
            _, idx, dist2 = kdtree.search_knn_vector_3d(p, 1)
            distances.append(np.sqrt(dist2[0]))

        distances = np.array(distances)
        
        return distances
    
    def apply_gaussian_filter(self, pcd, sigma=0.02, radius = 0.05):
        kdtree = o3d.geometry.KDTreeFlann(pcd)
        points = np.asarray(pcd.points)
        smoothed_points = np.zeros_like(points)

        for i in range(len(points)):
            [_, idxs, _] = kdtree.search_radius_vector_3d(pcd.points[i], radius)
            neighbors = points[idxs]
            distances = np.linalg.norm(neighbors - points[i], axis=1)
            weights = np.exp(-distances**2 / (2 * sigma**2))
            weights /= np.sum(weights)
            smoothed_points[i] = np.sum(neighbors * weights[:, np.newaxis], axis=0)

        pcd.points = o3d.utility.Vector3dVector(smoothed_points)

    def apply_random_scale(self, pcd):
        scale_x = np.random.uniform(0.5, 4.0)
        scale_y = np.random.uniform(0.5, 4.0)
        scale_z = np.random.uniform(0.5, 4.0)

        stretch_matrix = np.array([
            [scale_x, 0, 0],
            [0, scale_y, 0],
            [0, 0, scale_z]
        ])

        points = np.asarray(pcd.points)
        points = points @ stretch_matrix.T

        pcd.points = o3d.utility.Vector3dVector(points)
    
    def apply_random_rotation(self, pcd):
        angle = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1]
        ])

        points = np.asarray(pcd.points)
        points = points @ rotation_matrix.T

        pcd.points = o3d.utility.Vector3dVector(points)

    def normalize_mesh(self, points):
        min_vals = points.min(axis=0)
        max_vals = points.max(axis=0)
        points = (points - min_vals) / (max_vals - min_vals)
        return points
            
    def __iter__(self):
        num_spheres = np.random.randint(self.config.generation.min_parts, self.config.generation.max_parts + 1)
        while True:
            structure_type = np.random.choice(['sphere', 'cuboid'])
            if structure_type == 'sphere':
                structure = lib_neural_acd.generate_sphere_structure(num_spheres)
            elif structure_type == 'cuboid':
                structure = lib_neural_acd.generate_cuboid_structure(num_spheres)


            
            #print(verts)
            if self.output_meshes:
                lib_neural_acd.preprocess(structure, 50.0, 0.001)
            pcd = get_point_cloud(structure, num_points=10000)
            distances = self.get_distances(np.asarray(pcd.points), structure.cut_verts)

            if self.config.generation.random_scale:
                self.apply_random_scale(pcd)
            if self.config.generation.random_rotation:
                self.apply_random_rotation(pcd)
            if self.config.generation.gaussian_filter:
                self.apply_gaussian_filter(pcd,sigma=0.1,radius=random.uniform(0.01, 0.15))
            
            points = np.asarray(pcd.points)

            if self.config.generation.normalize_mesh:
                points = self.normalize_mesh(points)

            mesh = trimesh.Trimesh(vertices=structure.vertices, faces=structure.triangles)
            curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, points, 0.02)

            #add curvature channel
            points = np.hstack((points, curvature[:, np.newaxis]))

            points = torch.tensor(points, dtype=torch.float32)
            distances = torch.tensor(distances, dtype=torch.float32)
            distances = 1 - (torch.clamp(distances, 0.01, 0.05)-0.01)*25
            # distances[distances != 0] = 1

            if self.output_meshes:
                yield points, distances, structure
            else:
                yield points, distances