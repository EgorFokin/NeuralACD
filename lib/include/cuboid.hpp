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
  bool is_similar(const std::vector<Cuboid> &parts, double threshold);
  bool does_intersect(const std::vector<Cuboid> &parts, double threshold);
};
inline bool check_aabb_collision(const Cuboid &part1, const Cuboid &part2,
                                 double eps = 1e-6) {
  return (part1.min[0] <= part2.max[0] - eps &&
          part1.max[0] >= part2.min[0] + eps &&
          part1.min[1] <= part2.max[1] - eps &&
          part1.max[1] >= part2.min[1] + eps &&
          part1.min[2] <= part2.max[2] - eps &&
          part1.max[2] >= part2.min[2] + eps);
}
} // namespace neural_acd