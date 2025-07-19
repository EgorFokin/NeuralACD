#include "mesh.hpp"
#include <QuickHull.hpp>
#include <algorithm>
#include <btConvexHullComputer.h>
#include <cmath>
#include <iostream>
#include <random>
#include <sobol.hpp>
#include <stdexcept>

namespace acd_gen {

std::random_device rd;
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

double Area(Vec3D p0, Vec3D p1, Vec3D p2) {
  return 0.5 * sqrt(pow(p1[0] * p0[1] - p2[0] * p0[1] - p0[0] * p1[1] +
                            p2[0] * p1[1] + p0[0] * p2[1] - p1[0] * p2[1],
                        2) +
                    pow(p1[0] * p0[2] - p2[0] * p0[2] - p0[0] * p1[2] +
                            p2[0] * p1[2] + p0[0] * p2[2] - p1[0] * p2[2],
                        2) +
                    pow(p1[1] * p0[2] - p2[1] * p0[2] - p0[1] * p1[2] +
                            p2[1] * p1[2] + p0[1] * p2[2] - p1[1] * p2[2],
                        2));
}

Mesh::Mesh() {}

void Mesh::ComputeCH(Mesh &convex) const {
  /* fast convex hull algorithm */
  bool flag = true;
  quickhull::QuickHull<float> qh; // Could be double as well
  vector<quickhull::Vector3<float>> pointCloud;
  // Add vertices to point cloud
  for (int i = 0; i < (int)vertices.size(); i++) {
    pointCloud.push_back(quickhull::Vector3<float>(
        vertices[i][0], vertices[i][1], vertices[i][2]));
  }

  auto hull = qh.getConvexHull(pointCloud, true, false, flag);
  if (!flag) {
    // backup convex hull algorithm, stable but slow
    ComputeVCH(convex);
    return;
  }
  const auto &indexBuffer = hull.getIndexBuffer();
  const auto &vertexBuffer = hull.getVertexBuffer();
  for (int i = 0; i < (int)vertexBuffer.size(); i++) {
    convex.vertices.push_back(
        {vertexBuffer[i].x, vertexBuffer[i].y, vertexBuffer[i].z});
  }
  for (int i = 0; i < (int)indexBuffer.size(); i += 3) {
    convex.triangles.push_back({(int)indexBuffer[i + 2],
                                (int)indexBuffer[i + 1], (int)indexBuffer[i]});
  }
}

void Mesh::ComputeVCH(Mesh &convex) const {
  btConvexHullComputer ch;
  ch.compute(vertices, -1.0, -1.0);
  for (int32_t v = 0; v < ch.vertices.size(); v++) {
    convex.vertices.push_back(
        {ch.vertices[v].getX(), ch.vertices[v].getY(), ch.vertices[v].getZ()});
  }
  const int32_t nt = ch.faces.size();
  for (int32_t t = 0; t < nt; ++t) {
    const btConvexHullComputer::Edge *sourceEdge = &(ch.edges[ch.faces[t]]);
    int32_t a = sourceEdge->getSourceVertex();
    int32_t b = sourceEdge->getTargetVertex();
    const btConvexHullComputer::Edge *edge = sourceEdge->getNextEdgeOfFace();
    int32_t c = edge->getTargetVertex();
    while (c != a) {
      convex.triangles.push_back({(int)a, (int)b, (int)c});
      edge = edge->getNextEdgeOfFace();
      b = c;
      c = edge->getTargetVertex();
    }
  }
}

void Mesh::ExtractPointSet(std::vector<Vec3D> &samples,
                           std::vector<int> &sample_tri_ids, unsigned int seed,
                           size_t resolution, double base, bool flag,
                           Plane plane) {
  if (resolution == 0)
    return;
  double aObj = 0;
  for (int i = 0; i < (int)triangles.size(); i++) {
    aObj += Area(vertices[triangles[i][0]], vertices[triangles[i][1]],
                 vertices[triangles[i][2]]);
  }

  if (base != 0)
    resolution = size_t(max(1000, int(resolution * (aObj / base))));

  for (int i = 0; i < (int)triangles.size(); i++) {
    if (flag && plane.Side(vertices[triangles[i][0]], 1e-3) == 0 &&
        plane.Side(vertices[triangles[i][1]], 1e-3) == 0 &&
        plane.Side(vertices[triangles[i][2]], 1e-3) == 0) {
      continue;
    }
    double area = Area(vertices[triangles[i][0]], vertices[triangles[i][1]],
                       vertices[triangles[i][2]]);
    int N;
    if ((size_t)triangles.size() > resolution && resolution)
      N = max(int(i % ((int)triangles.size() / resolution) == 0),
              int(resolution / aObj * area));
    else
      N = max(int(i % 2 == 0), int(resolution / aObj * area));
    N = max(N, 1); // Ensure at least one sample per triangle
    std::uniform_int_distribution<int> seeder(0, 1000);
    int seed = seeder(rd);
    float r[2];
    for (int k = 0; k < N; k++) {
      double a, b;
      if (k % 3 == 0) {
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        //// random sample
        a = uniform(rd);
        b = uniform(rd);
      } else {
        //// quasirandom sample
        i4_sobol(2, &seed, r);
        a = r[0];
        b = r[1];
      }

      Vec3D v;
      v[0] = (1 - sqrt(a)) * vertices[triangles[i][0]][0] +
             (sqrt(a) * (1 - b)) * vertices[triangles[i][1]][0] +
             b * sqrt(a) * vertices[triangles[i][2]][0];
      v[1] = (1 - sqrt(a)) * vertices[triangles[i][0]][1] +
             (sqrt(a) * (1 - b)) * vertices[triangles[i][1]][1] +
             b * sqrt(a) * vertices[triangles[i][2]][1];
      v[2] = (1 - sqrt(a)) * vertices[triangles[i][0]][2] +
             (sqrt(a) * (1 - b)) * vertices[triangles[i][1]][2] +
             b * sqrt(a) * vertices[triangles[i][2]][2];
      samples.push_back(v);
      sample_tri_ids.push_back(i);
    }
  }
}

void Mesh::Clear() {
  vertices.clear();
  triangles.clear();
}

bool ComputeOverlapFace(Mesh &convex1, Mesh &convex2, Plane &plane) {
  bool flag;
  for (int i = 0; i < (int)convex1.triangles.size(); i++) {
    Plane p;
    Vec3D p1, p2, p3;
    p1 = convex1.vertices[convex1.triangles[i][0]];
    p2 = convex1.vertices[convex1.triangles[i][1]];
    p3 = convex1.vertices[convex1.triangles[i][2]];
    double a =
        (p2[1] - p1[1]) * (p3[2] - p1[2]) - (p2[2] - p1[2]) * (p3[1] - p1[1]);
    double b =
        (p2[2] - p1[2]) * (p3[0] - p1[0]) - (p2[0] - p1[0]) * (p3[2] - p1[2]);
    double c =
        (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]);
    p.a = a / sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
    p.b = b / sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
    p.c = c / sqrt(pow(a, 2) + pow(b, 2) + pow(c, 2));
    p.d = 0 - (p.a * p1[0] + p.b * p1[1] + p.c * p1[2]);

    short side1 = 0;
    for (int j = 0; j < (int)convex1.vertices.size(); j++) {
      short s = p.Side(convex1.vertices[j], 1e-8);
      if (s != 0) {
        side1 = s;
        flag = 1;
        break;
      }
    }

    for (int j = 0; j < (int)convex2.vertices.size(); j++) {
      short s = p.Side(convex2.vertices[j], 1e-8);
      if (!flag || s == side1) {
        flag = 0;
        break;
      }
    }
    if (flag) {
      plane = p;
      return true;
    }
  }
  return false;
}

void ExtractPointSet(Mesh &convex1, Mesh &convex2, std::vector<Vec3D> &samples,
                     std::vector<int> &sample_tri_ids, unsigned int seed,
                     size_t resolution) {
  std::vector<Vec3D> samples1, samples2;
  std::vector<int> sample_tri_ids1, sample_tri_ids2;
  double a1 = 0, a2 = 0;
  for (int i = 0; i < (int)convex1.triangles.size(); i++)
    a1 += Area(convex1.vertices[convex1.triangles[i][0]],
               convex1.vertices[convex1.triangles[i][1]],
               convex1.vertices[convex1.triangles[i][2]]);
  for (int i = 0; i < (int)convex2.triangles.size(); i++)
    a2 += Area(convex2.vertices[convex2.triangles[i][0]],
               convex2.vertices[convex2.triangles[i][1]],
               convex2.vertices[convex2.triangles[i][2]]);

  Plane overlap_plane;
  bool flag = ComputeOverlapFace(convex1, convex2, overlap_plane);

  convex1.ExtractPointSet(samples1, sample_tri_ids1, seed,
                          size_t(a1 / (a1 + a2) * resolution), 1, flag,
                          overlap_plane);
  convex2.ExtractPointSet(samples2, sample_tri_ids2, seed,
                          size_t(a2 / (a1 + a2) * resolution), 1, flag,
                          overlap_plane);

  samples.insert(samples.end(), samples1.begin(), samples1.end());
  samples.insert(samples.end(), samples2.begin(), samples2.end());

  sample_tri_ids.insert(sample_tri_ids.end(), sample_tri_ids1.begin(),
                        sample_tri_ids1.end());
  int N = (int)convex1.triangles.size();
  for (int i = 0; i < (int)sample_tri_ids2.size(); i++)
    sample_tri_ids.push_back(sample_tri_ids2[i] + N);
}
} // namespace acd_gen