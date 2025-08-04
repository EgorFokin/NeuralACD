#pragma once

#include <array>
#include <core.hpp>
#include <string>
#include <unordered_map>

namespace neural_acd {

class Icosphere : public Mesh {
public:
  double r;
  Icosphere() {};
  Icosphere(double r) : Icosphere(r, {0, 0, 0}) {};
  Icosphere(double r, Vec3D pos, int subdivisions = 3);
  void filter_cut_verts(std::vector<Icosphere> &parts, double eps = 1e-2);
  bool does_intersect(const std::vector<Icosphere> &parts,
                      double threshold = 0.0) const;

private:
  void create_icosahedron();
  void subdivide();
};
} // namespace neural_acd