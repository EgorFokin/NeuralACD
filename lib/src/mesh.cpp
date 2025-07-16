#include "mesh.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace acd_gen {
Vec3D operator+(const Vec3D &a, const Vec3D &b) {
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}
Vec3D operator-(const Vec3D &a, const Vec3D &b) {
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}
Vec3D operator*(const Vec3D &v, double scalar) {
    return {v[0] * scalar, v[1] * scalar, v[2] * scalar};
}
Vec3D operator/(const Vec3D &v, double scalar) {
    if (scalar == 0) {
        throw std::runtime_error("Division by zero in vector division");
    }
    return {v[0] / scalar, v[1] / scalar, v[2] / scalar};
}

double vector_length(const Vec3D &v) {
    return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

Vec3D normalize_vector(const Vec3D &v) {
    double length = vector_length(v);
    if (length == 0) {
        throw std::runtime_error("Cannot normalize a zero-length vector");
    }
    return {v[0] / length, v[1] / length, v[2] / length};
}

Vec3D slerp(const Vec3D &a, const Vec3D &b, double t) {
    double dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    dot = std::clamp(dot, -1.0, 1.0); // Ensure dot product is in valid range
    double theta = std::acos(dot) * t;
    Vec3D relative_vec = b - a * dot;
    relative_vec = normalize_vector(relative_vec);
    return a * std::cos(theta) + relative_vec * std::sin(theta);
}

Mesh::Mesh() {}
} // namespace acd_gen