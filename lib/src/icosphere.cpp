#include <cmath>
#include <icosphere.hpp>
#include <limits>
#include <mesh.hpp>

namespace acd_gen {

Icosphere::Icosphere(double r, Vec3D pos, int subdivisions) {
    this->r = r;
    this->pos = pos;
    create_icosahedron();
    for (int i = 0; i < subdivisions; ++i)
        subdivide();

    for (auto &v : vertices) {
        v[0] = v[0] * r + pos[0];
        v[1] = v[1] * r + pos[1];
        v[2] = v[2] * r + pos[2];
    }
}

void Icosphere::create_icosahedron() {
    const float X = 0.525731;
    const float Z = 0.850651;

    vertices = {
        {-X, 0, Z}, {X, 0, Z},  {-X, 0, -Z}, {X, 0, -Z},
        {0, Z, X},  {0, Z, -X}, {0, -Z, X},  {0, -Z, -X},
        {Z, X, 0},  {-Z, X, 0}, {Z, -X, 0},  {-Z, -X, 0},
    };

    for (auto &v : vertices) {
        normalize_vector(v);
    }

    triangles = {{0, 4, 1},  {0, 9, 4},  {9, 5, 4},  {4, 5, 8},  {4, 8, 1},
                 {8, 10, 1}, {8, 3, 10}, {5, 3, 8},  {5, 2, 3},  {2, 7, 3},
                 {7, 10, 3}, {7, 6, 10}, {7, 11, 6}, {11, 0, 6}, {0, 1, 6},
                 {6, 1, 10}, {9, 0, 11}, {9, 11, 2}, {9, 2, 5},  {7, 2, 11}};
}

void Icosphere::subdivide() {
    std::vector<std::array<int, 3>> new_tris;

    for (const auto &tri : triangles) {
        Vec3D v1 = slerp(vertices[tri[0]], vertices[tri[1]], 0.5);
        Vec3D v2 = slerp(vertices[tri[1]], vertices[tri[2]], 0.5);
        Vec3D v3 = slerp(vertices[tri[2]], vertices[tri[0]], 0.5);

        vertices.push_back(v1);
        vertices.push_back(v2);
        vertices.push_back(v3);

        int v1_index = vertices.size() - 3;
        int v2_index = vertices.size() - 2;
        int v3_index = vertices.size() - 1;

        new_tris.push_back({tri[0], v1_index, v3_index});
        new_tris.push_back({tri[1], v2_index, v1_index});
        new_tris.push_back({tri[2], v3_index, v2_index});
        new_tris.push_back({v1_index, v2_index, v3_index});
    }

    triangles = std::move(new_tris);
}

void Icosphere::filter_cut_verts(std::vector<Icosphere> &parts, double eps) {
    for (int i = cut_verts.size() - 1; i >= 0; --i) {
        const auto &vert = cut_verts[i];
        bool found = false;
        for (const auto &part : parts) {
            if (vector_length(part.pos - vert) < part.r - eps) {
                found = true;
                break;
            }
        }
        if (found) {
            cut_verts.erase(cut_verts.begin() + i);
        }
    }
}
} // namespace acd_gen