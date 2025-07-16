#include <cuboid.hpp>
#include <decompose_cuboids.hpp>
#include <decompose_spheres.hpp>
#include <generate.hpp>
#include <icosphere.hpp>
#include <mesh.hpp>
#include <random>

std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> dist(0.0, 1.0);

namespace acd_gen {

std::vector<Cuboid> test_cuboids = {
    Cuboid(0.8, 0.8, 0.8, {0.1, 0.0, 0.0}), Cuboid(1, 1, 0.1, {0, 0.2, 0.2}),
    Cuboid(1, 0.1, 1, {0, 0.3, 0.3}),
    // Cuboid(0.3, 0.5, 0.1, {0.0, 0.1, 0.3}),
    //  Cuboid(0.5, 0.5, 0.5, {0.0, 0.5, 0.0}),
    //  Cuboid(0.5, 0.5, 0.5, {0.5, 0.5, 0.0}),
};

Cuboid generate_cuboid() {
  double width = 0.05 + (0.5 - 0.05) * dist(gen); // Random between 0.05 and 0.5
  double height = 0.05 + (0.5 - 0.05) * dist(gen);
  double depth = 0.05 + (0.5 - 0.05) * dist(gen);
  double x = (1 - width) * dist(gen);
  double y = (1 - height) * dist(gen);
  double z = (1 - depth) * dist(gen);
  Vec3D pos = {x, y, z};

  double num = dist(gen);

  Cuboid cuboid(width, height, depth, pos);
  return cuboid;
}

Mesh generate_cuboid_structure(int obj_num) {
  std::vector<Cuboid> parts = std::vector<Cuboid>();
  for (int i = 0; i < obj_num; ++i) {
    Cuboid mesh = generate_cuboid();
    update_decomposition(parts, mesh);
  }

  merge_adjacent_cuboids(parts);

  for (int i = 0; i < parts.size(); ++i) {
    for (int j = 0; j < parts.size(); ++j) {
      if (i == j)
        continue; // Skip self-collision
      if (check_aabb_collision(parts[i], parts[j], 0))
        parts[i].compute_cut_quads(parts[j]);
    }
  }

  for (auto &part : parts) {
    part.remove_inner_part();
    part.filter_cut_verts(parts, 1e-2);
  }

  Mesh structure;
  int vert_offset = 0;
  for (const auto &part : parts) {
    structure.vertices.insert(structure.vertices.end(), part.vertices.begin(),
                              part.vertices.end());
    for (const auto &triangle : part.triangles) {
      structure.triangles.push_back({triangle[0] + vert_offset,
                                     triangle[1] + vert_offset,
                                     triangle[2] + vert_offset});
    }
    structure.cut_verts.insert(structure.cut_verts.end(),
                               part.cut_verts.begin(), part.cut_verts.end());

    vert_offset += part.vertices.size();
  }
  return structure;
}

//-----------------------------------------------------------

Icosphere generate_sphere(double min_radius, double max_radius) {
  double radius = dist(gen) * 0.5 + 0.1;
  Vec3D pos = {radius + (1 - radius) * dist(gen),
               radius + (1 - radius) * dist(gen),
               radius + (1 - radius) * dist(gen)};
  return Icosphere(radius, pos, 3);
}

Mesh generate_sphere_structure(int obj_num, double min_radius,
                               double max_radius) {
  std::vector<Icosphere> parts;
  for (int i = 0; i < obj_num; ++i) {
    Icosphere sphere = generate_sphere(min_radius, max_radius);
    update_decomposition(parts, sphere);
  }

  Mesh structure;
  int vert_offset = 0;
  for (const auto &part : parts) {
    structure.vertices.insert(structure.vertices.end(), part.vertices.begin(),
                              part.vertices.end());
    for (const auto &triangle : part.triangles) {
      structure.triangles.push_back({triangle[0] + vert_offset,
                                     triangle[1] + vert_offset,
                                     triangle[2] + vert_offset});
    }
    structure.cut_verts.insert(structure.cut_verts.end(),
                               part.cut_verts.begin(), part.cut_verts.end());

    vert_offset += part.vertices.size();
  }
  return structure;
}

Mesh test() {
  Cuboid cub = Cuboid(0.5, 0.5, 0.5, {0.01, 0.01, 0.0});
  std::vector<std::array<Vec3D, 4>> quad = {{{{{0.1, 0.1, 0.0}},
                                              {{0.3, 0.1, 0.0}},
                                              {{0.3, 0.3, 0.0}},
                                              {{0.1, 0.3, 0.0}}}},
                                            {{{{0.31, 0.31, 0.0}},
                                              {{0.45, 0.31, 0.0}},
                                              {{0.45, 0.45, 0.0}},
                                              {{0.31, 0.45, 0.0}}}}};
  cub.cut_face("-", 2, quad);
  Mesh mesh;
  mesh.vertices = cub.vertices;
  mesh.triangles = cub.triangles;
  return mesh;
}

void set_seed(unsigned int seed) { gen.seed(seed); }

} // namespace acd_gen