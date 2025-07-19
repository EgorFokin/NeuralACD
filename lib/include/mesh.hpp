#pragma once

#include <array>
#include <cmath>
#include <vector>

namespace acd_gen {

#define INF std::numeric_limits<double>::max()
using Vec3D = std::array<double, 3>;
Vec3D operator+(const Vec3D &a, const Vec3D &b);
Vec3D operator-(const Vec3D &a, const Vec3D &b);
Vec3D operator*(const Vec3D &v, double scalar);
Vec3D operator/(const Vec3D &v, double scalar);

double vector_length(const Vec3D &v);
Vec3D normalize_vector(const Vec3D &v);

Vec3D slerp(const Vec3D &a, const Vec3D &b, double t);

class Plane {
public:
  double a, b, c, d;
  bool pFlag;       // whether three point form exists
  Vec3D p0, p1, p2; // three point form
  short CutSide(Vec3D p0, Vec3D p1, Vec3D p2, Plane plane);
  short BoolSide(Vec3D p);
  short Side(Vec3D p, double eps = 1e-6);
  bool IntersectSegment(Vec3D p1, Vec3D p2, Vec3D &pi, double eps = 1e-6);
  Plane() { pFlag = false; };
  Plane(double _a, double _b, double _c, double _d) {
    a = _a;
    b = _b;
    c = _c;
    d = _d;
    pFlag = false;
  }
};

class Mesh {
public:
  Vec3D pos;
  std::vector<Vec3D> vertices;
  std::vector<std::array<int, 3>> triangles;
  std::vector<Vec3D> cut_verts;
  Mesh();
  void ComputeCH(Mesh &convex) const;
  void ComputeVCH(Mesh &convex) const;
  void ExtractPointSet(std::vector<Vec3D> &samples,
                       std::vector<int> &sample_tri_ids, unsigned int seed,
                       size_t resolution, double base = 0.0, bool flag = false,
                       Plane plane = Plane());
  void Clear();
};
using MeshList = std::vector<Mesh>;

inline Vec3D CrossProduct(Vec3D v, Vec3D w) {
  Vec3D res;
  res[0] = v[1] * w[2] - v[2] * w[1];
  res[1] = v[2] * w[0] - v[0] * w[2];
  res[2] = v[0] * w[1] - v[1] * w[0];

  return res;
}

inline double DotProduct(Vec3D v, Vec3D w) {
  return v[0] * w[0] + v[1] * w[1] + v[2] * w[2];
}

inline Vec3D CalFaceNormal(Vec3D p1, Vec3D p2, Vec3D p3) {
  Vec3D v, w, n, normal;
  v[1] = p2[1] - p1[1];
  v[2] = p2[2] - p1[2];
  w[0] = p3[0] - p1[0];
  w[1] = p3[1] - p1[1];
  w[2] = p3[2] - p1[2];

  n = CrossProduct(v, w);

  normal[0] = n[0] / sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2));
  normal[1] = n[1] / sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2));
  normal[2] = n[2] / sqrt(pow(n[0], 2) + pow(n[1], 2) + pow(n[2], 2));

  return normal;
}

inline short Plane::CutSide(Vec3D p0, Vec3D p1, Vec3D p2, Plane plane) {
  Vec3D normal = CalFaceNormal(p0, p1, p2);
  if (normal[0] * plane.a > 0 || normal[1] * plane.b > 0 ||
      normal[2] * plane.c > 0)
    return -1;
  return 1;
}

inline short Plane::BoolSide(Vec3D p) {
  double res = p[0] * a + p[1] * b + p[2] * c + d;
  if (res > 0)
    return 1;
  else
    return -1;
}

inline short Plane::Side(Vec3D p, double eps) {
  double res = p[0] * a + p[1] * b + p[2] * c + d;
  if (res > eps)
    return 1;
  else if (res < -1 * eps)
    return -1;
  return 0;
}

inline bool Plane::IntersectSegment(Vec3D p1, Vec3D p2, Vec3D &pi, double eps) {
  pi[0] =
      (p1[0] * b * p2[1] + p1[0] * c * p2[2] + p1[0] * d - p2[0] * b * p1[1] -
       p2[0] * c * p1[2] - p2[0] * d) /
      (a * p2[0] - a * p1[0] + b * p2[1] - b * p1[1] + c * p2[2] - c * p1[2]);
  pi[1] =
      (a * p2[0] * p1[1] + c * p1[1] * p2[2] + p1[1] * d - a * p1[0] * p2[1] -
       c * p1[2] * p2[1] - p2[1] * d) /
      (a * p2[0] - a * p1[0] + b * p2[1] - b * p1[1] + c * p2[2] - c * p1[2]);
  pi[2] =
      (a * p2[0] * p1[2] + b * p2[1] * p1[2] + p1[2] * d - a * p1[0] * p2[2] -
       b * p1[1] * p2[2] - p2[2] * d) /
      (a * p2[0] - a * p1[0] + b * p2[1] - b * p1[1] + c * p2[2] - c * p1[2]);

  if (std::min(p1[0] - eps, p2[0] - eps) <= pi[0] &&
      pi[0] <= std::max(p1[0] + eps, p2[0] + eps) &&
      std::min(p1[1] - eps, p2[1] - eps) <= pi[1] &&
      pi[1] <= std::max(p1[1] + eps, p2[1] + eps) &&
      std::min(p1[2] - eps, p2[2] - eps) <= pi[2] &&
      pi[2] <= std::max(p1[2] + eps, p2[2] + eps))
    return true;
  return false;
}

void ExtractPointSet(Mesh &convex1, Mesh &convex2, std::vector<Vec3D> &samples,
                     std::vector<int> &sample_tri_ids, unsigned int seed,
                     size_t resolution);

} // namespace acd_gen