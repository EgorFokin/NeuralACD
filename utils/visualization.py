import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import imageio
import os


os.environ["QT_QPA_PLATFORM"] = "xcb"


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
