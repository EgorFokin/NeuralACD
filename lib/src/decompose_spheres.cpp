#include <cmath>
#include <decompose_spheres.hpp>
#include <icosphere.hpp>
#include <iostream>
#include <mesh.hpp>
#include <tuple>
#include <vector>

namespace acd_gen {

std::vector<Icosphere *> detect_collisiont(std::vector<Icosphere> &parts,
                                           Icosphere &new_part) {
    std::vector<Icosphere *> colliding_parts;
    for (auto &part : parts) {
        if (vector_length(part.pos - new_part.pos) < part.r + new_part.r) {
            colliding_parts.push_back(&part);
        }
    }
    return colliding_parts;
}

int plane_side(Vec3D &point, Vec3D &normal, Vec3D &plane_point) {
    double d = (point[0] - plane_point[0]) * normal[0] +
               (point[1] - plane_point[1]) * normal[1] +
               (point[2] - plane_point[2]) * normal[2];
    if (d >= 0) {
        return 1; // Point is in front of the plane
    } else {
        return 0; // Point is behind the plane
    }
}

Vec3D get_intersect_with_plane(Vec3D &point1, Vec3D &point2, Vec3D &normal,
                               Vec3D &plane_point) {
    Vec3D line_dir = point2 - point1;
    double t = ((plane_point[0] - point1[0]) * normal[0] +
                (plane_point[1] - point1[1]) * normal[1] +
                (plane_point[2] - point1[2]) * normal[2]) /
               (line_dir[0] * normal[0] + line_dir[1] * normal[1] +
                line_dir[2] * normal[2]);
    return point1 + line_dir * t;
}

Icosphere part_after_intersection(Icosphere &old_part, Vec3D &direction,
                                  Vec3D &collision_center, bool side) {
    Icosphere new_part;
    new_part.pos = old_part.pos;
    new_part.r = old_part.r;
    new_part.cut_verts = old_part.cut_verts;

    for (auto tri : old_part.triangles) {
        Vec3D v1 = old_part.vertices[tri[0]];
        Vec3D v2 = old_part.vertices[tri[1]];
        Vec3D v3 = old_part.vertices[tri[2]];

        int s1 = plane_side(v1, direction, collision_center);
        int s2 = plane_side(v2, direction, collision_center);
        int s3 = plane_side(v3, direction, collision_center);

        // Flip the sides depending on which side of the plane the sphere is
        s1 = side ? s1 : 1 - s1;
        s2 = side ? s2 : 1 - s2;
        s3 = side ? s3 : 1 - s3;

        int sum = s1 + s2 + s3;

        if (sum == 0) {
            // All points are behind the plane
            continue;
        } else if (sum == 3) {
            // All points are in front of the plane
            new_part.vertices.push_back(v1);
            new_part.vertices.push_back(v2);
            new_part.vertices.push_back(v3);
            new_part.triangles.push_back({(int)new_part.vertices.size() - 3,
                                          (int)new_part.vertices.size() - 2,
                                          (int)new_part.vertices.size() - 1});
        } else if (sum == 2) {
            // Two points are in front of the plane, one is behind
            Vec3D new_v1, new_v2;
            Vec3D lone_vertex, v_other1, v_other2;
            if (!s1) {
                lone_vertex = v1;
                v_other1 = v2;
                v_other2 = v3;
            } else if (!s2) {
                lone_vertex = v2;
                v_other1 = v1;
                v_other2 = v3;
            } else {
                lone_vertex = v3;
                v_other1 = v1;
                v_other2 = v2;
            }

            // Compute intersection points
            new_v1 = get_intersect_with_plane(lone_vertex, v_other1, direction,
                                              collision_center);
            new_v2 = get_intersect_with_plane(lone_vertex, v_other2, direction,
                                              collision_center);

            // Add triangle: v_other1, new_v1, new_v2
            new_part.vertices.push_back(v_other1);
            new_part.vertices.push_back(new_v1);
            new_part.vertices.push_back(new_v2);
            new_part.triangles.push_back({(int)new_part.vertices.size() - 3,
                                          (int)new_part.vertices.size() - 2,
                                          (int)new_part.vertices.size() - 1});

            // Add triangle: new_v2, v_other2, new_v1
            new_part.vertices.push_back(v_other2);
            new_part.triangles.push_back({(int)new_part.vertices.size() - 1,
                                          (int)new_part.vertices.size() - 4,
                                          (int)new_part.vertices.size() - 2});
            new_part.cut_verts.push_back(new_v1);
            new_part.cut_verts.push_back(new_v2);

        } else {
            // One point is in front of the plane, two are behind
            Vec3D new_v1, new_v2;
            Vec3D front_vertex, v_other1, v_other2;
            if (s1) {
                front_vertex = v1;
                v_other1 = v2;
                v_other2 = v3;
            } else if (s2) {
                front_vertex = v2;
                v_other1 = v1;
                v_other2 = v3;
            } else {
                front_vertex = v3;
                v_other1 = v1;
                v_other2 = v2;
            }
            // Compute intersection points
            new_v1 = get_intersect_with_plane(front_vertex, v_other1, direction,
                                              collision_center);
            new_v2 = get_intersect_with_plane(front_vertex, v_other2, direction,
                                              collision_center);
            // Add triangle: front_vertex, new_v1, new_v2
            new_part.vertices.push_back(front_vertex);
            new_part.vertices.push_back(new_v1);
            new_part.vertices.push_back(new_v2);
            new_part.triangles.push_back({(int)new_part.vertices.size() - 3,
                                          (int)new_part.vertices.size() - 2,
                                          (int)new_part.vertices.size() - 1});
            new_part.cut_verts.push_back(new_v1);
            new_part.cut_verts.push_back(new_v2);
        }
    }
    return new_part;
}

std::pair<Icosphere, Icosphere> handle_two_colliding_parts(Icosphere part1,
                                                           Icosphere &part2) {
    double d = vector_length(part1.pos - part2.pos);
    Vec3D direction = normalize_vector(part2.pos - part1.pos);
    double d_center =
        (part1.r * part1.r - part2.r * part2.r + d * d) / (2.0 * d);
    Vec3D collision_center = part1.pos + direction * d_center;

    bool s1 = plane_side(part1.pos, direction, collision_center);
    bool s2 = plane_side(part2.pos, direction, collision_center);
    bool side;
    if (s1 == s2) {
        side = (vector_length(part1.pos - collision_center) <
                vector_length(part2.pos - collision_center)) &
               !s1;
    } else
        side = s1;

    Icosphere new_part1 =
        part_after_intersection(part1, direction, collision_center, side);
    Icosphere new_part2 =
        part_after_intersection(part2, direction, collision_center, !side);

    return std::make_pair(new_part1, new_part2);
}

void update_decomposition(std::vector<Icosphere> &parts, Icosphere &new_part) {
    std::vector<Icosphere *> colliding_parts =
        detect_collisiont(parts, new_part);

    if (colliding_parts.empty()) {
        // No collisions detected, simply add the new part
        parts.push_back(new_part);
        return;
    }

    std::vector<Icosphere> new_parts;

    // check if one of the spheres is completely inside another
    for (int i = colliding_parts.size() - 1; i >= 0; --i) {
        Icosphere *colliding_part = colliding_parts[i];

        if (vector_length(colliding_part->pos - new_part.pos) <=
            new_part.r - colliding_part->r) {
            parts.erase(parts.begin() + (colliding_part - parts.data()));
            update_decomposition(parts,
                                 new_part); // recalculate collisions
            return;
        }

        if (vector_length(colliding_part->pos - new_part.pos) <=
            colliding_part->r - new_part.r) {
            return;
        }
    }

    for (int i = colliding_parts.size() - 1; i >= 0; --i) {
        Icosphere *colliding_part = colliding_parts[i];

        Icosphere new_part1, new_part2;
        std::tie(new_part1, new_part2) =
            handle_two_colliding_parts(*colliding_part, new_part);

        // Remove the colliding part from the parts vector
        parts.erase(parts.begin() + (colliding_part - parts.data()));

        new_part = new_part2;
        new_parts.push_back(new_part1);
    }

    for (auto &part : new_parts) {
        parts.push_back(part);
    }

    parts.push_back(new_part);

    for (auto &part : parts) {
        part.filter_cut_verts(parts);
    }
}
} // namespace acd_gen