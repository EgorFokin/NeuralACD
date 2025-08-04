#include <config.hpp>
#include <core.hpp>
#include <cuboid.hpp>
#include <decompose_cuboids.hpp>
#include <decompose_spheres.hpp>
#include <generate.hpp>
#include <icosphere.hpp>
#include <preprocess.hpp>
#include <random>

namespace neural_acd {

std::uniform_real_distribution<> dist(0.0, 1.0);

Cuboid generate_cuboid(double min_width, double max_width) {

  double width = min_width + (max_width - min_width) * dist(random_engine);
  double height = min_width + (max_width - min_width) * dist(random_engine);
  double depth = min_width + (max_width - min_width) * dist(random_engine);
  double x = (1 - width) * dist(random_engine);
  double y = (1 - height) * dist(random_engine);
  double z = (1 - depth) * dist(random_engine);
  Vec3D pos = {x, y, z};

  double num = dist(random_engine);

  Cuboid cuboid(width, height, depth, pos);
  return cuboid;
}

Mesh generate_cuboid_structure(int obj_num) {
  std::vector<Cuboid> parts = std::vector<Cuboid>();
  for (int i = 0; i < obj_num; ++i) {
    Cuboid mesh = generate_cuboid(config.generation_cuboid_width_min,
                                  config.generation_cuboid_width_max);
    if (mesh.is_similar(parts, 0.05) ||
        !(mesh.does_intersect(parts, 0) || parts.size() == 0)) {
      i--;
      continue;
    }

    update_decomposition(parts, mesh);
  }

  if (parts.size() == 0) {
    return generate_cuboid_structure(obj_num); // Retry if no parts generated
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
  double radius = min_radius + (max_radius - min_radius) * dist(random_engine);
  Vec3D pos = {radius + (1 - 2 * radius) * dist(random_engine),
               radius + (1 - 2 * radius) * dist(random_engine),
               radius + (1 - 2 * radius) * dist(random_engine)};
  return Icosphere(radius, pos, config.generation_icosphere_subdivs);
}

Mesh generate_sphere_structure(int obj_num) {
  std::vector<Icosphere> parts;
  for (int i = 0; i < obj_num; ++i) {
    Icosphere sphere = generate_sphere(config.generation_sphere_radius_min,
                                       config.generation_sphere_radius_max);

    if (parts.size() != 0 && !sphere.does_intersect(parts, 0)) {
      i--;
      continue;
    }
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

} // namespace neural_acd