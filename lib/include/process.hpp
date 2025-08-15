#pragma once

#include <core.hpp>

namespace neural_acd {
std::vector<Plane> get_best_planes(std::vector<Vec3D> cut_points);
MeshList process(Mesh mesh, std::vector<Vec3D> cut_points,
                 std::string stats_file = "");

} // namespace neural_acd