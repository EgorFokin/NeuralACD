import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..","lib", "build"))

from omegaconf import OmegaConf
import lib_neural_acd
import numpy as np
import open3d as o3d


def get_lib_mesh(mesh):
    vertices = lib_neural_acd.VecArray3d(mesh.vertices.tolist())
    triangles = lib_neural_acd.make_vecarray3i(mesh.faces.tolist())
    mesh =  lib_neural_acd.Mesh()
    
    mesh.vertices = vertices
    mesh.triangles = triangles
    return mesh

def normalize_mesh(mesh):
    vertices = np.asarray(mesh.vertices)
    centroid = np.mean(vertices, axis=0)
    vertices -= centroid
    max_distance = np.max(np.linalg.norm(vertices, axis=1))
    vertices /= max_distance
    mesh.vertices = lib_neural_acd.VecArray3d(vertices.tolist())
    return mesh

def get_point_cloud(mesh, num_points=10000):
    o3d_mesh = o3d.geometry.TriangleMesh()

    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)  
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.triangles)

    pcd = o3d_mesh.sample_points_uniformly(number_of_points=10000)
    return pcd

def load_lib_config(config):
    lib_config = lib_neural_acd.config

    lib_config.generation_cuboid_width_max = config.lib.generation.cuboid.width_max
    lib_config.generation_cuboid_width_min = config.lib.generation.cuboid.width_min
    lib_config.generation_sphere_radius_min = config.lib.generation.sphere.radius_min
    lib_config.generation_sphere_radius_max = config.lib.generation.sphere.radius_max
    lib_config.generation_icosphere_subdivs = config.lib.generation.sphere.subdivs

    lib_config.pcd_res = config.lib.pcd_res

    lib_config.remesh_res = config.lib.remesh.resolution
    lib_config.remesh_threshold = config.lib.remesh.threshold

    lib_config.cost_rv_k = config.lib.cost_rv_k

    lib_config.merge_threshold = config.lib.merge_threshold

    lib_config.jlinkage_sigma = config.lib.jlinkage.sigma
    lib_config.jlinkage_num_samples = config.lib.jlinkage.num_samples
    lib_config.jlinkage_threshold = config.lib.jlinkage.threshold
    lib_config.jlinkage_outlier_threshold = config.lib.jlinkage.outlier_threshold

def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)

    load_lib_config(conf)
    return conf