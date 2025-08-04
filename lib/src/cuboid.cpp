#include <CDT.h>
#include <CDTUtils.h>
#include <cmath>
#include <core.hpp>
#include <cuboid.hpp>
#include <iostream>
#include <limits>
#include <unordered_map>

namespace neural_acd {

Cuboid::Cuboid(double width, double height, double depth, Vec3D pos) {
  this->pos = pos;
  vertices = {{0, 0, 0},
              {width, 0, 0},
              {width, height, 0},
              {0, height, 0},
              {0, 0, depth},
              {width, 0, depth},
              {width, height, depth},
              {0, height, depth}};

  for (auto &vertex : vertices) {
    vertex[0] += pos[0];
    vertex[1] += pos[1];
    vertex[2] += pos[2];
  }

  // compute min and max
  min = {INF, INF, INF};
  max = {-INF, -INF, -INF};

  for (const auto &vertex : vertices) {
    min[0] = std::min(min[0], vertex[0]);
    min[1] = std::min(min[1], vertex[1]);
    min[2] = std::min(min[2], vertex[2]);
    max[0] = std::max(max[0], vertex[0]);
    max[1] = std::max(max[1], vertex[1]);
    max[2] = std::max(max[2], vertex[2]);
  }

  triangles = {
      {2, 1, 0}, {0, 3, 2}, // Bottom
      {4, 5, 6}, {6, 7, 4}, // Top
      {0, 1, 5}, {5, 4, 0}, // Front
      {2, 3, 7}, {7, 6, 2}, // Back
      {1, 2, 6}, {6, 5, 1}, // Right
      {3, 0, 4}, {4, 7, 3}  // Left
  };
}

void Cuboid::get_side_quad(std::string dir, int dim,
                           std::array<int, 4> &verts_i,
                           std::array<int, 2> &tris_i) {
  if (dir == "-" && dim == 0) {
    verts_i = {0, 4, 7, 3}; // should be in quad order
    tris_i = {10, 11};      // Left
  } else if (dir == "-" && dim == 1) {
    verts_i = {0, 1, 5, 4};
    tris_i = {4, 5}; // Front
  } else if (dir == "-" && dim == 2) {
    verts_i = {0, 1, 2, 3};
    tris_i = {0, 1}; // Bottom
  } else if (dir == "+" && dim == 0) {
    verts_i = {1, 2, 6, 5};
    tris_i = {8, 9}; // Right
  } else if (dir == "+" && dim == 1) {
    verts_i = {2, 3, 7, 6};
    tris_i = {6, 7}; // Back
  } else if (dir == "+" && dim == 2) {
    verts_i = {4, 5, 6, 7};
    tris_i = {2, 3}; // Top
  }
}

void Cuboid::update_side(std::string dir, int dim, double value) {
  std::array<int, 4> verts_i;
  std::array<int, 2> tris_i;
  get_side_quad(dir, dim, verts_i, tris_i);

  for (int i = 0; i < 4; ++i) {
    vertices[verts_i[i]][dim] = value;
  }
  if (dir == "-")
    min[dim] = value;
  else
    max[dim] = value;
}

bool Cuboid::is_similar(const std::vector<Cuboid> &parts, double threshold) {
  for (const auto &p : parts) {
    if (!check_aabb_collision(p, *this, -threshold)) {
      continue; // not intersecting
    }
    for (int dim = 0; dim < 3; ++dim) {
      if (std::abs(p.min[dim] - min[dim]) < threshold ||
          std::abs(p.max[dim] - max[dim]) < threshold ||
          std::abs(p.min[dim] - max[dim]) < threshold ||
          std::abs(p.max[dim] - min[dim]) < threshold) {
        return true;
      }
    }
  }
  return false;
}

bool Cuboid::does_intersect(const std::vector<Cuboid> &parts,
                            double threshold) {
  for (const auto &p : parts) {
    if (check_aabb_collision(p, *this, threshold)) {

      return true;
    }
  }
  return false;
}

bool check_aabb_collision(Cuboid &part, const Vec3D &vert, double eps) {
  // Check if the vertex is within the AABB of the part
  return (part.min[0] + eps <= vert[0] && part.max[0] - eps >= vert[0] &&
          part.min[1] + eps <= vert[1] && part.max[1] - eps >= vert[1] &&
          part.min[2] + eps <= vert[2] && part.max[2] - eps >= vert[2]);
}

// precondition: part collides with this cuboid
void Cuboid::compute_cut_quads(Cuboid &part, double eps) {
  Vec3D v1, v2, v3, v4;
  bool skip_d1_mn = 0, skip_d2_mn = 0, skip_d1_mx = 0, skip_d2_mx = 0;
  bool found = false;

  for (int dim = 0; dim < 3; ++dim) {
    int d1 = (dim + 1) % 3;
    int d2 = (dim + 2) % 3;
    if (std::abs(part.max[dim] - min[dim]) < eps ||
        std::abs(part.min[dim] - max[dim]) < eps) {

      double fixed = (std::abs(part.max[dim] - min[dim]) < eps)
                         ? min[dim]
                         : max[dim]; // fixed dimension

      int d1 = (dim + 1) % 3;
      int d2 = (dim + 2) % 3;

      // Calculate cut
      double min_d1 = std::max(part.min[d1], min[d1]) + eps;
      double max_d1 = std::min(part.max[d1], max[d1]) - eps;
      double min_d2 = std::max(part.min[d2], min[d2]) + eps;
      double max_d2 = std::min(part.max[d2], max[d2]) - eps;

      if ((std::abs(min_d1 - max_d1) < 1e-3 ||
           std::abs(min_d2 - max_d2) < 1e-3)) {
        // edges probably touch, not adjacent
        continue;
      }

      // check if the touch by side
      if (std::abs(part.min[d1] - min[d1]) < eps) {
        skip_d1_mn = true;
      }
      if (std::abs(part.max[d1] - max[d1]) < eps) {
        skip_d1_mx = true;
      }
      if (std::abs(part.min[d2] - min[d2]) < eps) {
        skip_d2_mn = true;
      }
      if (std::abs(part.max[d2] - max[d2]) < eps) {
        skip_d2_mx = true;
      }

      Vec3D base = {0.0, 0.0, 0.0};
      base[dim] = fixed;

      v1 = base; // bottom left
      v1[d1] = min_d1;
      v1[d2] = min_d2;
      v2 = base; // bottom right
      v2[d1] = max_d1;
      v2[d2] = min_d2;
      v3 = base; // top right
      v3[d1] = max_d1;
      v3[d2] = max_d2;
      v4 = base; // top left
      v4[d1] = min_d1;
      v4[d2] = max_d2;

      found = true;

      break; // only one face can match
    }
  }

  if (!found)
    return;

  std::array<Vec3D, 4> quad = {v1, v2, v3,
                               v4}; // order is important for cut_verts
  cut_quads.push_back(quad);

  std::vector<Vec3D> new_vertices;
  if (!skip_d2_mn)
    subdivide_edge(v1, v2, new_vertices, 4);
  if (!skip_d1_mx)
    subdivide_edge(v2, v3, new_vertices, 4);
  if (!skip_d2_mx)
    subdivide_edge(v3, v4, new_vertices, 4);
  if (!skip_d1_mn)
    subdivide_edge(v4, v1, new_vertices, 4);

  for (const auto &new_vertex : new_vertices) {
    cut_verts.push_back(new_vertex);
  }
}

void Cuboid::filter_cut_verts(std::vector<Cuboid> &parts, double eps) {
  for (int i = cut_verts.size() - 1; i >= 0; --i) {
    const auto &vert = cut_verts[i];
    int found = 0; // Number of parts that are adjacent to this vertex
    for (auto &part : parts) {
      if (&part == this)
        continue; // Skip self
      if (!check_aabb_collision(part, vert, -1e-6))
        continue;
      if (std::abs(part.max[0] - vert[0]) < eps ||
          std::abs(part.min[0] - vert[0]) < eps ||
          std::abs(part.max[1] - vert[1]) < eps ||
          std::abs(part.min[1] - vert[1]) < eps ||
          std::abs(part.max[2] - vert[2]) < eps ||
          std::abs(part.min[2] - vert[2]) < eps) {
        found++;
      }
    }
    if (found >= 2) {
      cut_verts.erase(cut_verts.begin() + i);
    }
  }
}
void Cuboid::remove_inner_part() {
  std::unordered_map<std::string,
                     std::unordered_map<int, std::vector<std::array<Vec3D, 4>>>>
      quads; // quads[dir][dim] = quads for that face
  for (const auto &quad : cut_quads) {
    for (int dim = 0; dim < 3; ++dim) {
      if (std::abs(quad[0][dim] - quad[1][dim]) < 1e-6 &&
          std::abs(quad[0][dim] - quad[2][dim]) < 1e-6 &&
          std::abs(quad[0][dim] - quad[3][dim]) < 1e-6) {

        std::string dir =
            (std::abs(quad[0][dim] - min[dim]) < 1e-6) ? "-" : "+";
        quads[dir][dim].push_back(quad);
      }
    }
  }

  for (auto &[dir, inner] : quads) {
    for (auto &[dim, quads_list] : inner) {
      if (quads_list.size() > 0) {
        cut_face(dir, dim, quads_list);
      }
    }
  }
}

void merge_quads(std::vector<std::array<Vec3D, 4>> &quads, int dim,
                 double eps = 1e-6) {
  int d1 = (dim + 1) % 3;
  int d2 = (dim + 2) % 3;

  for (size_t i = 0; i < quads.size(); ++i) {
    for (size_t j = i + 1; j < quads.size(); ++j) {

      double min_d1_i = quads[i][0][d1];
      double max_d1_i = quads[i][1][d1];
      double min_d2_i = quads[i][0][d2];
      double max_d2_i = quads[i][2][d2];
      double min_d1_j = quads[j][0][d1];
      double max_d1_j = quads[j][1][d1];
      double min_d2_j = quads[j][0][d2];
      double max_d2_j = quads[j][2][d2];

      if (std::abs(max_d1_i - min_d1_j) < eps &&
          std::abs(min_d2_i - min_d2_j) < eps &&
          std::abs(max_d2_i - max_d2_j) < eps) {
        quads[i][1][d1] = max_d1_j;
        quads[i][2][d1] = max_d1_j;
        quads.erase(quads.begin() + j);
        merge_quads(quads, dim);
        break;
      }
      if (std::abs(max_d1_j - min_d1_i) < eps &&
          std::abs(min_d2_j - min_d2_i) < eps &&
          std::abs(max_d2_j - max_d2_i) < eps) {
        quads[j][1][d1] = max_d1_i;
        quads[j][2][d1] = max_d1_i;
        quads.erase(quads.begin() + i);
        merge_quads(quads, dim);
        break;
      }
      if (std::abs(max_d2_i - min_d2_j) < eps &&
          std::abs(min_d1_i - min_d1_j) < eps &&
          std::abs(max_d1_i - max_d1_j) < eps) {
        quads[i][2][d2] = max_d2_j;
        quads[i][3][d2] = max_d2_j;
        quads.erase(quads.begin() + j);
        merge_quads(quads, dim);
        break;
      }
      if (std::abs(max_d2_j - min_d2_i) < eps &&
          std::abs(min_d1_j - min_d1_i) < eps &&
          std::abs(max_d1_j - max_d1_i) < eps) {
        quads[j][2][d2] = max_d2_i;
        quads[j][3][d2] = max_d2_i;
        quads.erase(quads.begin() + i);
        merge_quads(quads, dim);
        break;
      }
    }
  }
}

void Cuboid::cut_face(std::string dir, int dim,
                      std::vector<std::array<Vec3D, 4>> &quads) {

  CDT::Triangulation<double> cdt;

  merge_quads(quads, dim);

  int num_quads = quads.size();

  int d1 = (dim + 1) % 3;
  int d2 = (dim + 2) % 3;

  std::array<int, 4> verts_i;
  std::array<int, 2> tris_i;
  get_side_quad(dir, dim, verts_i, tris_i);

  Vec3D v1 = vertices[verts_i[0]];
  Vec3D v2 = vertices[verts_i[1]];
  Vec3D v3 = vertices[verts_i[2]];
  Vec3D v4 = vertices[verts_i[3]];

  std::vector<std::array<double, 2>> points;
  std::vector<std::pair<int, int>> edges;

  Vec3D face_normal = calc_face_normal(vertices[triangles[tris_i[0]][0]],
                                       vertices[triangles[tris_i[0]][1]],
                                       vertices[triangles[tris_i[0]][2]]);

  // add outer border
  points.push_back({v1[d1], v1[d2]});
  points.push_back({v2[d1], v2[d2]});
  points.push_back({v3[d1], v3[d2]});
  points.push_back({v4[d1], v4[d2]});

  edges.push_back({0, 1});
  edges.push_back({1, 2});
  edges.push_back({2, 3});
  edges.push_back({3, 0});

  // add inner quads
  for (auto &quad : quads) {
    points.push_back({quad[0][d1], quad[0][d2]});
    points.push_back({quad[1][d1], quad[1][d2]});
    points.push_back({quad[2][d1], quad[2][d2]});
    points.push_back({quad[3][d1], quad[3][d2]});
    edges.push_back({points.size() - 4, points.size() - 3});
    edges.push_back({points.size() - 3, points.size() - 2});
    edges.push_back({points.size() - 2, points.size() - 1});
    edges.push_back({points.size() - 1, points.size() - 4});
  }

  cdt.insertVertices(
      points.begin(), points.end(),
      [](const std::array<double, 2> &p) { return p[0]; },
      [](const std::array<double, 2> &p) { return p[1]; });
  cdt.insertEdges(
      edges.begin(), edges.end(),
      [](const std::pair<int, int> &e) { return e.first; },
      [](const std::pair<int, int> &e) { return e.second; });
  cdt.eraseSuperTriangle();

  int offset = vertices.size() - 4;

  auto map_to_3d = [&](const CDT::V2d<double> &p) {
    Vec3D v;
    v[d1] = p.x;
    v[d2] = p.y;
    if (dir == "-") {
      v[dim] = min[dim];
    } else {
      v[dim] = max[dim];
    }
    return v;
  };

  vertices[verts_i[0]] = map_to_3d(cdt.vertices[0]);
  vertices[verts_i[1]] = map_to_3d(cdt.vertices[1]);
  vertices[verts_i[2]] = map_to_3d(cdt.vertices[2]);
  vertices[verts_i[3]] = map_to_3d(cdt.vertices[3]);

  for (int i = 4; i < cdt.vertices.size(); i++) {
    vertices.push_back(map_to_3d(cdt.vertices[i]));
  }

  std::vector<std::array<int, 3>> new_tris;

  for (size_t i = 0; i < (size_t)cdt.triangles.size(); i++) {
    int i1 = cdt.triangles[i].vertices[0];
    int i2 = cdt.triangles[i].vertices[1];
    int i3 = cdt.triangles[i].vertices[2];

    // check if the triangle is inside one of the quads, delete if so
    bool skip = false;
    for (int quad_i = 0; quad_i < num_quads; ++quad_i) {
      if (i1 >= 4 + quad_i * 4 && i1 < 4 + (quad_i + 1) * 4 &&
          i2 >= 4 + quad_i * 4 && i2 < 4 + (quad_i + 1) * 4 &&
          i3 >= 4 + quad_i * 4 && i3 < 4 + (quad_i + 1) * 4) {
        skip = true;
        break;
      }
    }
    if (skip) {
      continue;
    }

    std::array<int, 4> remap = {verts_i[0], verts_i[1], verts_i[2], verts_i[3]};

    auto remap_index = [&](int i) {
      return (i >= 0 && i < 4) ? remap[i] : i + offset;
    };

    i1 = remap_index(i1);
    i2 = remap_index(i2);
    i3 = remap_index(i3);

    Vec3D tri_normal =
        calc_face_normal(vertices[i1], vertices[i2], vertices[i3]);

    if (dot(tri_normal, face_normal) < 1e-6) { // fix normals
      new_tris.push_back({i1, i3, i2});
    } else {
      new_tris.push_back({i1, i2, i3});
    }
  }

  // replace old triangles. Not deleting to not break get_side_quad
  triangles[tris_i[0]] = new_tris[0];
  triangles[tris_i[1]] = new_tris[1];

  for (size_t i = 2; i < new_tris.size(); ++i) {
    triangles.push_back(new_tris[i]);
  }
}

} // namespace neural_acd