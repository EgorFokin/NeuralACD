#pragma once

#include <cuboid.hpp>
#include <mesh.hpp>

namespace acd_gen {
void update_decomposition(std::vector<Cuboid> &parts, Cuboid &new_part);
void merge_adjacent_cuboids(std::vector<Cuboid> &parts, double eps = 1e-6);

} // namespace acd_gen