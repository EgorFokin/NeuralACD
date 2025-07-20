#pragma once

#include <array>
#include <core.hpp>
#include <string>
#include <unordered_map>

namespace neural_acd {
class Cuboid : public Mesh {
public:
  Vec3D min;
  Vec3D max;
  std::vector<std::array<Vec3D, 4>> cut_quads;
  Cuboid() : Cuboid(1.0, 1.0, 1.0, {0, 0, 0}) {};
  Cuboid(double width, double height, double depth)
      : Cuboid(width, height, depth, {0, 0, 0}) {};
  Cuboid(double width, double height, double depth,
         Vec3D pos); // pos is at the -x,-y,-z corner
  void get_side_quad(std::string dir, int dim, std::array<int, 4> &verts_i,
                     std::array<int, 2> &tris_i);
  void update_side(std::string dir, int dim, double value);
  void filter_cut_verts(std::vector<Cuboid> &parts, double eps = 1e-2);
  void compute_cut_quads(Cuboid &part, double eps = 1e-6);
  void remove_inner_part();
  void cut_face(std::string dir, int dim,
                std::vector<std::array<Vec3D, 4>> &quads);
};
bool check_aabb_collision(Cuboid &part1, Cuboid &part2, double eps = 1e-6);
} // namespace neural_acd