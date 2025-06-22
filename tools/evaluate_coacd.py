import coacd_modified
import os
import open3d as o3d
import datetime
import numpy as np


if __name__ == "__main__":
    VHACD = "data\\v_hacd"
    RANDOM_ROTATION = True
    start_time = datetime.datetime.now()
    num_parts = 0
    i = 0
    for files in os.listdir(VHACD):
        if files.endswith(".off"):
            mesh = o3d.io.read_triangle_mesh(os.path.join(VHACD, files))
            if RANDOM_ROTATION:
                rotation = mesh.get_rotation_matrix_from_xyz(np.random.rand(3) * 2 * np.pi)
                mesh.rotate(rotation, center=(0, 0, 0))
            cmesh = coacd_modified.Mesh(np.asarray(mesh.vertices), np.asarray(mesh.triangles))
            parts = coacd_modified.run_coacd(cmesh)
            num_parts += len(parts)
            print("processed: ", i+1)
            i+=1

    print("runtime: ", datetime.datetime.now() - start_time)
    print("average number of parts: ", num_parts/i)