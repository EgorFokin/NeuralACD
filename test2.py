import coacd_modified
import trimesh

if __name__ == "__main__":
    mesh = trimesh.load("mesh.obj")
    cmesh = coacd_modified.Mesh(mesh.vertices, mesh.faces)
    parts = []
    out = coacd_modified.run_coacd(cmesh)
    for vs, fs in out:
        parts.append(trimesh.Trimesh(vs, fs))
    scene = trimesh.Scene()
    for p in parts:
        scene.add_geometry(p)
    scene.export("decomposed.obj")

    planes = coacd_modified.best_cutting_planes(cmesh, num_planes=5)
    print([(plane.a, plane.b, plane.c, plane.d, plane.score) for plane in planes])

    # cmesh = coacd_modified2.Mesh(mesh.vertices, mesh.faces)

    # plane = coacd_modified2.best_cutting_plane(cmesh)

    # print(plane.a, plane.b, plane.c, plane.d)
    
