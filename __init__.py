import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__),"lib","build"))
print(sys.path)
import lib_neural_acd


import bpy
from bpy.props import IntProperty

bl_info = {
    "name": "ACDgen",
    "author": "Your Name",
    "version": (1, 0),
    "blender": (2, 80, 0),
    "location": "View3D > Sidebar > ACDgen",
    "description": "Generation of Convex Decomposed Meshes",
    "category": "Object",
}

class OBJECT_OT_generate_cuboids(bpy.types.Operator):
    bl_idname = "object.generate_cuboids"
    bl_label = "Generate a mesh from cuboids"
    bl_description = "Generate a mesh from cuboids"
    bl_options = {'REGISTER', 'UNDO'}

    num_cuboids: IntProperty(
        name="Number of Cuboids",
        default=10,
        min=1,
        max=1000,
        description="Number of cuboids to generate"
    ) # type: ignore

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        #Generate cuboid
        mesh = lib_neural_acd.generate_cuboid_structure(self.num_cuboids)
        if not mesh:
            self.report({'ERROR'}, "Failed to generate cuboids")
            return {'CANCELLED'}
        
        mesh_data = bpy.data.meshes.new("triangle_mesh")
        mesh_data.from_pydata(mesh.vertices, [], mesh.triangles)
        mesh_data.update()


        obj = bpy.data.objects.new("Part", mesh_data)

        # Link to scene collection
        bpy.context.collection.objects.link(obj)

        for cluster in mesh.cut_verts:
            # Create a new mesh for cut vertices
            cut_mesh_data = bpy.data.meshes.new("cut_verts_mesh")
            cut_mesh_data.from_pydata(cluster, [], [])
            cut_mesh_data.update()
            cut_obj = bpy.data.objects.new("CutVertsObject", cut_mesh_data)
            bpy.context.collection.objects.link(cut_obj)

        return {'FINISHED'}
    
class OBJECT_OT_generate_spheres(bpy.types.Operator):
    bl_idname = "object.generate_spheres"
    bl_label = "Generate a mesh from spheres"
    bl_description = "Generate a mesh from spheres"
    bl_options = {'REGISTER', 'UNDO'}

    num_spheres: IntProperty(
        name="Number of spheres",
        default=10,
        min=1,
        max=1000,
        description="Number of spheres to generate"
    ) # type: ignore

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)

    def execute(self, context):
        #Generate cuboid
        mesh = lib_neural_acd.generate_sphere_structure(self.num_spheres)
        if not mesh:
            self.report({'ERROR'}, "Failed to generate cuboids")
            return {'CANCELLED'}
        
        mesh_data = bpy.data.meshes.new("triangle_mesh")
        mesh_data.from_pydata(mesh.vertices, [], mesh.triangles)
        mesh_data.update()


        obj = bpy.data.objects.new("Part", mesh_data)

        # Link to scene collection
        bpy.context.collection.objects.link(obj)
        

        for cluster in mesh.cut_verts:
            # Create a new mesh for cut vertices
            cut_mesh_data = bpy.data.meshes.new("cut_verts_mesh")
            cut_mesh_data.from_pydata(list(cluster), [], [])
            cut_mesh_data.update()
            cut_obj = bpy.data.objects.new("CutVertsObject", cut_mesh_data)
            bpy.context.collection.objects.link(cut_obj)

        return {'FINISHED'}
    
class OBJECT_OT_test(bpy.types.Operator):
    bl_idname = "object.test"
    bl_label = "test"
    bl_description = "test"
    bl_options = {'REGISTER', 'UNDO'}


    def execute(self, context):
        #Generate cuboid
        mesh = lib_neural_acd.test()

        mesh_data = bpy.data.meshes.new("test_mesh")
        mesh_data.from_pydata(mesh.vertices, [],mesh.triangles)
        mesh_data.update()
        cut_obj = bpy.data.objects.new("TestMesh", mesh_data)
        bpy.context.collection.objects.link(cut_obj)

        return {'FINISHED'}

# UI Panel in the 3D Viewport sidebar
class OBJECT_PT_acd_gen(bpy.types.Panel):
    bl_label = "ACDgen Panel"
    bl_idname = "OBJECT_PT_acd_gen"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'ACDgen'

    def draw(self, context):
        layout = self.layout
        layout.operator("object.generate_cuboids")
        layout.operator("object.generate_spheres")
        layout.operator("object.test")

# Registration
classes = [OBJECT_OT_generate_cuboids,OBJECT_OT_generate_spheres,OBJECT_OT_test, OBJECT_PT_acd_gen]

def register():
    for cls in classes:
        bpy.utils.register_class(cls)

def unregister():
    for cls in reversed(classes):
        bpy.utils.unregister_class(cls)

