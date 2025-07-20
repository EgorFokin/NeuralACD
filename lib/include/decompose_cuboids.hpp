#pragma once

#include <core.hpp>
#include <cuboid.hpp>

namespace neural_acd {
void update_decomposition(std::vector<Cuboid> &parts, Cuboid &new_part);
void merge_adjacent_cuboids(std::vector<Cuboid> &parts, double eps = 1e-6);

} // namespace neural_acd