import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "..","lib", "build"))

from omegaconf import OmegaConf
import lib_neural_acd
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import trimesh

def set_seed(seed):
    np.random.seed(seed)
    lib_neural_acd.set_seed(seed)
    pl.seed_everything(seed)
    

def get_lib_mesh(mesh):
    vertices = lib_neural_acd.VecArray3d(mesh.vertices.tolist())
    triangles = lib_neural_acd.make_vecarray3i(mesh.faces.tolist())
    mesh =  lib_neural_acd.Mesh()
    
    mesh.vertices = vertices
    mesh.triangles = triangles
    return mesh

def normalize_mesh(mesh):
    verts = np.asarray(mesh.vertices)
    min_vals = verts.min(axis=0)
    centered = verts - min_vals
    scale = np.max(centered.max(axis=0))  # max range across all axes

    struct_vertices = (verts - min_vals) / scale
    mesh.vertices = lib_neural_acd.VecArray3d(struct_vertices)

def load_mesh(mesh, config):
    lib_mesh = get_lib_mesh(mesh)
    normalize_mesh(lib_mesh)
    lib_neural_acd.preprocess(lib_mesh, 50.0, 0.55)

    points = lib_neural_acd.VecArray3d()
    point_tris = lib_neural_acd.VecInt()
    lib_mesh.extract_point_set(points, point_tris, config.general.num_points)

    points = np.asarray(points)

    mesh = trimesh.Trimesh(vertices=np.asarray(lib_mesh.vertices), faces=np.asarray(lib_mesh.triangles))
    curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, points, radius=0.02)
    normals = mesh.face_normals[np.asarray(point_tris)]

    points = np.hstack((points, curvature[:, np.newaxis], normals))
    return lib_mesh, points



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

    lib_config.dbscan_eps = config.lib.dbscan.eps
    lib_config.dbscan_min_pts = config.lib.dbscan.min_pts
    lib_config.dbscan_outlier_threshold = config.lib.dbscan.outlier_threshold

    lib_config.jlinkage_sigma = config.lib.jlinkage.sigma
    lib_config.jlinkage_num_samples = config.lib.jlinkage.num_samples
    lib_config.jlinkage_threshold = config.lib.jlinkage.threshold
    lib_config.jlinkage_outlier_threshold = config.lib.jlinkage.outlier_threshold

    lib_config.refinement_iterations = config.lib.refinement_iterations

    lib_config.process_output_parts = config.lib.return_parts

def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)

    load_lib_config(conf)
    return conf