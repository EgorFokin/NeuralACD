from utils.BaseUtils import *
from decompose import decompose
import os
from concurrent.futures import ProcessPoolExecutor

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

def process(mesh_hash,vertices,faces):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(faces)
    parts = decompose(mesh,1)
    for i,part in enumerate(parts):
        o3d.io.write_triangle_mesh(f"data/ShapeNetParts/{mesh_hash}_{i}.obj", part)

if __name__ == "__main__":
    if not os.path.exists("data/ShapeNetParts"):
        os.makedirs("data/ShapeNetParts")
    
    executor = ProcessPoolExecutor()

    for mesh_hash,mesh in load_shapenet(debug=True,data_folder="data/ShapeNetRedistributed"):

        if os.path.exists(f"data/ShapeNetParts/{mesh_hash}_0.obj"):
            continue

        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)

        #clip function sometimes causes assertion fails, which cannot be caught, so we have to use multiprocessing to avoid errors
        #this is probably due to the generated plane not intersecting with the mesh

        
        executor.submit(process,mesh_hash,vertices,faces)
