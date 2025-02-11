import bpy
import mathutils

# Define the plane equation coefficients
a, b, c, d = tuple(map(float,"1.0 0.0 0.0 -0.049596263376138164".split()))


if a!=0:
    p1 = mathutils.Vector((-d/a,1,1))
    p2 = mathutils.Vector((-d/a,-1,1))
    p3 = mathutils.Vector((-d/a,1,-1))
    p4 = mathutils.Vector((-d/a,-1,-1))
elif b != 0 :
    p1 = mathutils.Vector((1,-d/b,1))
    p2 = mathutils.Vector((-1,-d/b,1))
    p3 = mathutils.Vector((1,-d/b,-1))
    p4 = mathutils.Vector((-1,-d/b,-1))
elif c != 0:
    p1 = mathutils.Vector((1,1,-d/c))
    p2 = mathutils.Vector((-1,1,-d/c))
    p3 = mathutils.Vector((1,-1,-d/c))
    p4 = mathutils.Vector((-1,-1,-d/c))

# Create a new mesh and object
mesh = bpy.data.meshes.new(name="QuadFace")
obj = bpy.data.objects.new(name="QuadFaceObject", object_data=mesh)

# Link the object to the current scene
bpy.context.collection.objects.link(obj)

# Define vertices and face
vertices = [p1, p2, p4, p3]
faces = [(0, 1, 2, 3)]

# Create the mesh
mesh.from_pydata(vertices, [], faces)
mesh.update()