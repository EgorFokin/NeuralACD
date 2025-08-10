#include "core.hpp"
#include <QuickHull.hpp>
#include <algorithm>
#include <boost/random/sobol.hpp>
#include <boost/random/uniform_01.hpp>
#include <boost/random/variate_generator.hpp>
#include <btConvexHullComputer.h>
#include <cmath>
#include <iostream>
#include <random>
#include <stdexcept>
#include <unordered_set>

namespace neural_acd {

boost::random::sobol sobol_engine(2);
boost::uniform_01<double> uniform_dist;
boost::variate_generator<boost::random::sobol &, boost::uniform_01<double>>
    sobol_gen(sobol_engine, uniform_dist);

void set_seed(unsigned int seed) { random_engine.seed(seed); }

Mesh::Mesh() {}

void Mesh::compute_ch(Mesh &convex) const {
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
    compute_vch(convex);
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

void Mesh::compute_vch(Mesh &convex) const {
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

void Mesh::extract_point_set(std::vector<Vec3D> &samples,
                             std::vector<int> &sample_tri_ids,
                             size_t resolution, double base, bool flag,
                             Plane plane, bool one_per_tri) {

  if (triangles.empty() || vertices.empty()) {
    return;
  }

  double aObj = 0.0;

  std::vector<double> areas = std::vector<double>(triangles.size(), 0.0);
  for (size_t i = 0; i < triangles.size(); i++) {
    double area =
        triangle_area(vertices[triangles[i][0]], vertices[triangles[i][1]],
                      vertices[triangles[i][2]]);
    areas[i] = area;
    aObj += area;
  }

  if (base != 0)
    resolution = size_t(max(1000, int(resolution * (aObj / base))));

  discrete_distribution<size_t> triangle_index_generator(areas.begin(),
                                                         areas.end());

  std::uniform_real_distribution<double> uniform_dist(0.0, 1.0);

  std::unordered_set<size_t> sampled_tris;

  int sampled = 0;

  while (sampled < resolution) {

    size_t tidx = triangle_index_generator(random_engine);

    const auto &tri = triangles[tidx];

    if (flag && plane.side(vertices[tri[0]], 1e-3) == 0 &&
        plane.side(vertices[tri[1]], 1e-3) == 0 &&
        plane.side(vertices[tri[2]], 1e-3) == 0) {
      continue;
    }

    double a = uniform_dist(random_engine);
    double b = uniform_dist(random_engine);

    Vec3D v;
    v[0] = (1 - sqrt(a)) * vertices[tri[0]][0] +
           (sqrt(a) * (1 - b)) * vertices[tri[1]][0] +
           b * sqrt(a) * vertices[tri[2]][0];
    v[1] = (1 - sqrt(a)) * vertices[tri[0]][1] +
           (sqrt(a) * (1 - b)) * vertices[tri[1]][1] +
           b * sqrt(a) * vertices[tri[2]][1];
    v[2] = (1 - sqrt(a)) * vertices[tri[0]][2] +
           (sqrt(a) * (1 - b)) * vertices[tri[1]][2] +
           b * sqrt(a) * vertices[tri[2]][2];
    samples.push_back(v);
    sample_tri_ids.push_back(tidx);
    sampled_tris.insert(tidx);
    sampled++;
  }

  if (one_per_tri) {
    for (size_t i = 0; i < triangles.size(); i++) {
      if (sampled_tris.find(i) == sampled_tris.end()) {

        const auto &tri = triangles[i];

        double a = uniform_dist(random_engine);
        double b = uniform_dist(random_engine);

        Vec3D v;
        v[0] = (1 - sqrt(a)) * vertices[tri[0]][0] +
               (sqrt(a) * (1 - b)) * vertices[tri[1]][0] +
               b * sqrt(a) * vertices[tri[2]][0];
        v[1] = (1 - sqrt(a)) * vertices[tri[0]][1] +
               (sqrt(a) * (1 - b)) * vertices[tri[1]][1] +
               b * sqrt(a) * vertices[tri[2]][1];
        v[2] = (1 - sqrt(a)) * vertices[tri[0]][2] +
               (sqrt(a) * (1 - b)) * vertices[tri[1]][2] +
               b * sqrt(a) * vertices[tri[2]][2];

        samples.push_back(v);
        sample_tri_ids.push_back(i);
      }
    }
  }
}

void Mesh::clear() {
  vertices.clear();
  triangles.clear();
}

void Mesh::normalize() {
  if (vertices.empty())
    return;

  Vec3D min = vertices[0], max = vertices[0];

  // Compute bounding box
  for (const auto &v : vertices) {
    for (int i = 0; i < 3; i++) {
      if (v[i] < min[i])
        min[i] = v[i];
      if (v[i] > max[i])
        max[i] = v[i];
    }
  }

  // Compute center of bounding box
  Vec3D center = {(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0,
                  (min[2] + max[2]) / 2.0};

  // Compute longest side length
  double scale = std::max({max[0] - min[0], max[1] - min[1], max[2] - min[2]});

  // Normalize: center and scale to fit in [-1, 1] along the longest axis
  for (auto &v : vertices) {
    v = (v - center) * 2 / scale;
  }
}

void Mesh::normalize(std::vector<Vec3D> &points) {
  if (vertices.empty())
    return;

  Vec3D min = vertices[0], max = vertices[0];

  // Compute bounding box
  for (const auto &v : vertices) {
    for (int i = 0; i < 3; i++) {
      if (v[i] < min[i])
        min[i] = v[i];
      if (v[i] > max[i])
        max[i] = v[i];
    }
  }

  // Compute center of bounding box
  Vec3D center = {(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0,
                  (min[2] + max[2]) / 2.0};

  // Compute longest side length
  double scale = std::max({max[0] - min[0], max[1] - min[1], max[2] - min[2]});

  // Normalize: center and scale to fit in [-1, 1] along the longest axis
  for (auto &v : vertices) {
    v = (v - center) * 2 / scale;
  }
  for (auto &p : points) {
    p = (p - center) * 2 / scale;
  }
}

void get_midpoint(const Vec3D &v1, const Vec3D &v2, Vec3D &mid) {
  mid[0] = (v1[0] + v2[0]) / 2.0;
  mid[1] = (v1[1] + v2[1]) / 2.0;
  mid[2] = (v1[2] + v2[2]) / 2.0;
}

void subdivide_edge(const Vec3D &v1, const Vec3D &v2,
                    std::vector<Vec3D> &new_vertices, int depth) {
  Vec3D mid;
  get_midpoint(v1, v2, mid);
  new_vertices.push_back(mid);
  if (depth == 0) {
    return;
  }
  subdivide_edge(v1, mid, new_vertices, depth - 1);
  subdivide_edge(mid, v2, new_vertices, depth - 1);
}

bool compute_overlap_face(Mesh &convex1, Mesh &convex2, Plane &plane) {
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
      short s = p.side(convex1.vertices[j], 1e-8);
      if (s != 0) {
        side1 = s;
        flag = 1;
        break;
      }
    }

    for (int j = 0; j < (int)convex2.vertices.size(); j++) {
      short s = p.side(convex2.vertices[j], 1e-8);
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

void extract_point_set(Mesh &convex1, Mesh &convex2,
                       std::vector<Vec3D> &samples,
                       std::vector<int> &sample_tri_ids, size_t resolution) {
  std::vector<Vec3D> samples1, samples2;
  std::vector<int> sample_tri_ids1, sample_tri_ids2;
  double a1 = 0, a2 = 0;
  for (int i = 0; i < (int)convex1.triangles.size(); i++)
    a1 += triangle_area(convex1.vertices[convex1.triangles[i][0]],
                        convex1.vertices[convex1.triangles[i][1]],
                        convex1.vertices[convex1.triangles[i][2]]);
  for (int i = 0; i < (int)convex2.triangles.size(); i++)
    a2 += triangle_area(convex2.vertices[convex2.triangles[i][0]],
                        convex2.vertices[convex2.triangles[i][1]],
                        convex2.vertices[convex2.triangles[i][2]]);

  Plane overlap_plane;
  bool flag = compute_overlap_face(convex1, convex2, overlap_plane);

  convex1.extract_point_set(samples1, sample_tri_ids1,
                            size_t(a1 / (a1 + a2) * resolution), 1, flag,
                            overlap_plane);
  convex2.extract_point_set(samples2, sample_tri_ids2,
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

void LoadingBar::step() {
  string bar;
  bar += "\r" + message + " [";
  int pos = (current_step * bar_length) / total_steps;
  for (int i = 0; i < bar_length; ++i) {
    if (i < pos)
      bar += "=";
    else
      bar += " ";
  }
  bar +=
      "] " + std::to_string(current_step) + "/" + std::to_string(total_steps);
  std::cout << bar;
  std::cout.flush();
  current_step++;
}

void LoadingBar::finish() {
  std::cout << "\r" + message + " [";
  for (int i = 0; i < bar_length; ++i) {
    if (i < bar_length)
      std::cout << "=";
    else
      std::cout << " ";
  }
  std::cout << "] " << total_steps << "/" << total_steps << std::endl;
}
} // namespace neural_acd