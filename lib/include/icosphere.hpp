#pragma once

#include <array>
#include <mesh.hpp>
#include <string>
#include <unordered_map>

namespace acd_gen {

class Icosphere : public Mesh {
  public:
    double r;
    Icosphere() {};
    Icosphere(double r) : Icosphere(r, {0, 0, 0}) {};
    Icosphere(double r, Vec3D pos, int subdivisions = 3);
    void filter_cut_verts(std::vector<Icosphere> &parts, double eps = 1e-2);

  private:
    void create_icosahedron();
    void subdivide();
};
} // namespace acd_gen