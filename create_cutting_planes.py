
import open3d as o3d
import numpy as np
from matplotlib import cm
import trimesh
import torch
import sys
import os

sys.path.append("lib/build")


from utils.jlinkage import JLinkage

from sklearn.cluster import DBSCAN

from utils.ACDgen import ACDgen

from model.model import ACDModel
from scipy.spatial import cKDTree


CHECKPOINT="checkpoints/20,07,2025-11:53:02/best-model-ema_loss=0.21378971636295319.ckpt"

def normalize_points(pcd):
    points = np.asarray(pcd.points)
    centroid = np.mean(points, axis=0)
    points -= centroid
    max_distance = np.max(np.linalg.norm(points, axis=1))
    points /= max_distance
    pcd.points = o3d.utility.Vector3dVector(points)

def get_trimesh_plane(a, b, c, d):
    # Normalize the normal
    normal = np.array([a, b, c], dtype=np.float64)
    normal = normal / np.linalg.norm(normal)

    # Find a point on the plane that's inside or near the [0,1]^3 box
    # We'll use the projection of the center of the box (0.5, 0.5, 0.5) onto the plane
    center = np.array([0.5, 0.5, 0.5])
    distance = np.dot(normal, center) + d
    point_on_plane = center - distance * normal

    # Create a small quad (plane proxy) within [0,1]^3
    size = 4  # slightly larger than the box to fill view
    plane = trimesh.creation.box(extents=(size, size, 0.005))  # very thin
    plane.apply_translation(-plane.centroid)

    # Rotate plane to match normal
    T = trimesh.geometry.align_vectors([0, 0, 1], normal)
    if T is not None:
        plane.apply_transform(T)

    # Translate to the projected point
    plane.apply_translation(point_on_plane)

    # Set visual style
    plane.visual.face_colors = [255, 0, 0, 100]  # red, semi-transparent

    return plane


def remove_outliers(points, threshold=0.05):
    # Remove points that have less than 3 neighbors within the threshold
    
    tree = cKDTree(points)
    neighbors = tree.query_ball_point(points, threshold)
    inliers = [i for i, n in enumerate(neighbors) if len(n) >= 3]
    return points[inliers]


def split_points(points, threshold=0.03, iterations=10000):
    best_eq = None
    best_inliers = None
    best_score = -1e9


    N = points.shape[0]
    for i in range(iterations):
        sample = points[np.random.choice(N, 3, replace=False)]
        p1, p2, p3 = sample
        
        # 2. Compute the plane normal
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        if np.linalg.norm(normal) == 0:
            continue  # skip degenerate triplets

        normal = normal / np.linalg.norm(normal)
        a, b, c = normal
        d = -np.dot(normal, p1)

        # 3. Compute distance of all points to the plane
        distances = np.abs((points @ normal) + d)

        # 4. Count inliers
        inliers = distances < threshold
        num_inliers = np.sum(inliers)

        # clustering = DBSCAN(eps=0.1, min_samples=10).fit(points[inliers])
        # if clustering.labels_.max() > 0 or sum(clustering.labels_ == -1) > sum(clustering.labels_ >= 0):
        #     continue

        score = num_inliers# - 0.1*num_inliers_not_cut

        # 5. Update best model
        if score > best_score:
            best_inliers = inliers
            best_eq = (a, b, c, d)
            best_score = score

    return get_trimesh_plane(*best_eq), best_inliers
        



it = ACDgen(output_meshes=True).__iter__()
lib_acd_gen.set_seed(42)
#next(it)  
points, distances_t, structure = next(it)

points = np.asarray(points)


# o3d_mesh = o3d.io.read_triangle_mesh("data/meshes/cow.obj")

# pcd = o3d_mesh.sample_points_uniformly(number_of_points=10000)
# normalize_points(pcd)

# points =np.asarray(pcd.points) 



with torch.no_grad():
    model = ACDModel().cuda()
    model.load_state_dict(torch.load(CHECKPOINT)["state_dict"])
    model.eval()
    distances = model(torch.tensor(points, dtype=torch.float32).cuda().unsqueeze(0))
    distances = torch.sigmoid(distances)  # Ensure distances are in [0, 1] range
    distances = distances.squeeze().cpu().numpy()
    print(max(distances))

# distances = distances_t

threshold = 0.7
distances[distances < threshold] = 0
distances[distances >= threshold] = 1

cut_points = points[distances == 1]
cut_points = remove_outliers(cut_points, threshold=0.05)

print(f"Cut points: {cut_points.shape[0]}")

colormap = cm.get_cmap("jet")
colors = colormap(distances)[:, :3]  # RGB, invert to make close = red


point_cloud = trimesh.points.PointCloud(points, colors=colors)
point_cloud.show()


# scene = trimesh.Scene()

# for mesh in decomposed:
#     triangles = np.asarray(mesh.triangles)
#     vertices = np.asarray(mesh.vertices)
#     if (triangles.shape[0] == 0 or vertices.shape[0] == 0):
#         continue
#     tmesh = trimesh.Trimesh(vertices=vertices, faces=triangles, process=True)

#     #apply a random color to each mesh
#     tmesh.visual.face_colors = np.random.randint(0, 255, (3), dtype=np.uint8)
    
#     scene.add_geometry(tmesh)

# scene.export("decomposed.glb")

# tmesh = trimesh.Trimesh(vertices=structure.vertices, faces=structure.triangles, process=True)
# tmesh.export("structure.glb")

# scene.show()
