#pragma once
#include <core.hpp>
#include <cuboid.hpp>
#include <icosphere.hpp>

namespace neural_acd {

Cuboid generate_cuboid(double min_width, double max_width);
Icosphere generate_sphere(double min_radius, double max_radius);
Mesh generate_cuboid_structure(int obj_num);
Mesh generate_sphere_structure(int obj_num);
} // namespace neural_acd