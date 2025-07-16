#pragma once
#include <cuboid.hpp>
#include <icosphere.hpp>
#include <mesh.hpp>

namespace acd_gen {

Cuboid generate_cuboid();
Icosphere generate_sphere(double min_radius = 0.1, double max_radius = 0.5);
Mesh generate_cuboid_structure(int obj_num);
Mesh generate_sphere_structure(int obj_num, double min_radius = 0.1,
                               double max_radius = 0.5);
Mesh test();
void set_seed(unsigned int seed);
} // namespace acd_gen