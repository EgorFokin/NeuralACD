import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os
import pyrender
import trimesh
from PIL import Image
import seaborn as sns


os.environ["QT_QPA_PLATFORM"] = "xcb"

xR = trimesh.transformations.rotation_matrix(np.radians(20), [1, 0, 0])
yR = trimesh.transformations.rotation_matrix(np.radians(-30), [0, 1, 0])

R = np.dot(xR, yR)



def save_rotating_pcd_gif(points, distances, gif_path="pcd_rotation.gif", frames=36):
    distances = np.asarray(distances, dtype=np.float32)
    points = points[:, :3]
    norm = plt.Normalize(vmin=np.min(distances), vmax=np.max(distances))
    base_colors = plt.get_cmap("jet")(norm(distances))[:, :3]

    # Blend positive distances with orange
    orange = np.array([1.0, 0.5, 0.0])
    brightness = np.clip(distances, 0, 1)[:, None]
    colors = base_colors * (1 - 0.5 * brightness) + orange * (0.5 * brightness)

    # Set alpha based on distance
    alpha = np.where(distances > 0, 1, 0.5)[:, None]
    rgba = np.hstack([colors, alpha])

    # Store generated frames
    images = []

    # Rotation loop
    for angle in np.linspace(0, 360, frames, endpoint=False):
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.view_init(elev=30, azim=angle)  # change view

        ax.scatter(points[:, 0], points[:, 1], points[:, 2],
                   c=np.clip(rgba[:, :3], 0, 1), s=2, alpha=rgba[:, 3])

        ax.set_axis_off()
        ax.set_box_aspect([1, 1, 1])

        # Save to buffer
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images.append(image)

        plt.close(fig)

    # Save GIF
    imageio.mimsave(gif_path, images, fps=15)
    print(f"GIF saved to {gif_path}")

def show_pcd(points, values,clusters=None):
    values = np.asarray(values, dtype=np.float32)
    points = points[:, :3]


    if clusters is None:
        cmap = sns.color_palette("magma", as_cmap=True)
        norm = plt.Normalize(vmin=np.min(values), vmax=np.max(values))
        colors = cmap(norm(values))

        trimesh.points.PointCloud(points, colors=colors).show()
    else:
        colored = points[values == 1]
        colors = np.zeros((len(colored), 3))
        print(len(clusters))
        for cluster in clusters:
            cluster_color = np.random.rand(3)
            colors[cluster] = cluster_color
        
        trimesh.points.PointCloud(colored, colors=colors).show()
        
        
def render_pcd(points, values, file):
    """
    Renders a point cloud with colors based on values and saves it to a file.
    """
    points = np.asarray(points)
    values = np.asarray(values)

    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)

    # Center points
    centroid = (bbox_min + bbox_max) / 2
    points_centered = points - centroid

    rotation_mat = R[:3,:3].copy()
    # rotation_mat[:,0] *= -1
    points = points_centered @ rotation_mat.T

    max_extent = np.max(bbox_max - bbox_min) / 2.0

    # Scale mesh so max extent fits in 1 unit (so box fits in [-1, 1])
    scale_factor = 1.0 / max_extent

    points *= scale_factor * 1.3


    cmap = sns.color_palette("magma", as_cmap=True)
    norm = plt.Normalize(vmin=np.min(values), vmax=np.max(values))
    colors = cmap(norm(values))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(-points[:, 0], points[:, 2], points[:, 1], c=colors, s=1) #swap y and z for visualization
    
    ax.set_axis_off()
    ax.view_init(elev=0, azim=90)

    
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.set_box_aspect([1,1,1])

    # plt.show()
    plt.savefig(file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def render_scene(scene, output_path: str, resolution=(800, 800)):
    """
    Render a trimesh.Scene using pyrender and save it as a PNG.

    Parameters
    ----------
    scene
        The scene to render.
    output_path : str
        Path to save the rendered PNG file.
    resolution : tuple
        (width, height) in pixels for the output image.
    """

    if isinstance(scene, trimesh.Trimesh):
        # Convert trimesh to pyrender scene
        scene = pyrender.Scene()
        mesh = pyrender.Mesh.from_trimesh(scene)
        scene.add(mesh)

    # Convert trimesh scene to pyrender scene
    pyrender_scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

    bbox_min, bbox_max = scene.bounds
    center = (bbox_min + bbox_max) / 2


    max_extent = np.max(bbox_max - bbox_min) / 2.0

    # Scale mesh so max extent fits in 1 unit (so box fits in [-1, 1])
    scale_factor = 1.0 / max_extent



    for geom_name, mesh in scene.geometry.items():

        # If no colors, give it a default gray
        if mesh.visual.face_colors is None or len(mesh.visual.face_colors) == 0:
            mesh.visual.face_colors = [200, 200, 200, 255]

        mesh.vertices -= center

        mesh.vertices *= scale_factor

        mesh.apply_transform(R)


        pyr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        pyrender_scene.add(pyr_mesh)


    # Add a camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

    cam_pose = np.eye(4)
    cam_pose[2, 3] = 2.5  # Move the camera back

    pyrender_scene.add(camera, pose=cam_pose)

    # Add a light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    pyrender_scene.add(light, pose=cam_pose)

    # Render offscreen
    r = pyrender.OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])
    color, _ = r.render(pyrender_scene)

    # Save to file
    Image.fromarray(color).save(output_path)
    r.delete()
