import bpy
import mathutils

def create_plane_from_equation(a, b, c, d, size=2):
    """
    Creates a plane in Blender based on the plane equation ax + by + cz + d = 0.

    Args:
        a, b, c, d (float): Plane equation coefficients.
        size (float): Scale of the plane.
    """

    # Normal vector of the plane
    normal = mathutils.Vector((a, b, c))

    # Ensure the normal is not zero
    if normal.length == 0:
        raise ValueError("Invalid normal vector (0,0,0), cannot define a plane.")

    # Compute a point on the plane (assuming c ≠ 0)
    if c != 0:
        point = mathutils.Vector((0, 0, -d / c))
    elif b != 0:
        point = mathutils.Vector((0, -d / b, 0))
    else:
        point = mathutils.Vector((-d / a, 0, 0))

    # Create a Blender plane mesh
    bpy.ops.mesh.primitive_plane_add(size=size, location=(0, 0, 0))
    plane_obj = bpy.context.object

    # Rotate plane to align with normal
    up = mathutils.Vector((0, 0, 1))  # Default Blender plane normal
    rotation_quat = up.rotation_difference(normal)
    plane_obj.rotation_euler = rotation_quat.to_euler()

    # Move plane to align with the computed point
    plane_obj.location = point

    return plane_obj

# Example usage: Create a plane from the equation x + 2y + 3z - 5 = 0
create_plane_from_equation(-0.7082102220603499, 0.1683554188623689, -0.6856345486546795, 0.14679409509425312, size=4)
