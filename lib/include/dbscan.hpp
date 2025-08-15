#pragma once

#include <cassert>
#include <core.hpp>
#include <cstddef>
#include <cstdlib>
#include <span>
#include <vector>

namespace neural_acd {
std::vector<std::vector<size_t>> dbscan(const std::span<const Vec3D> &data,
                                        double eps, int min_pts);

}