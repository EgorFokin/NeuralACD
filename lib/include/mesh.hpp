#pragma once

#include <array>
#include <cmath>
#include <vector>

namespace acd_gen {
using Vec3D = std::array<double, 3>;
Vec3D operator+(const Vec3D &a, const Vec3D &b);
Vec3D operator-(const Vec3D &a, const Vec3D &b);
Vec3D operator*(const Vec3D &v, double scalar);
Vec3D operator/(const Vec3D &v, double scalar);

double vector_length(const Vec3D &v);
Vec3D normalize_vector(const Vec3D &v);

Vec3D slerp(const Vec3D &a, const Vec3D &b, double t);

class Mesh {
  public:
    Vec3D pos;
    std::vector<Vec3D> vertices;
    std::vector<std::array<int, 3>> triangles;
    std::vector<Vec3D> cut_verts;
    Mesh();
};

using MeshList = std::vector<Mesh>;

} // namespace acd_gen