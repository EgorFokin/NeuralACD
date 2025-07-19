#pragma once

#include <mesh.hpp>

namespace acd_gen {
MeshList process(Mesh mesh, std::vector<Vec3D> cut_points);

inline int32_t FindMinimumElement(const std::vector<double> d, double *const m,
                                  const int32_t begin, const int32_t end) {
  int32_t idx = -1;
  double min = (std::numeric_limits<double>::max)();
  for (size_t i = begin; i < size_t(end); ++i) {
    if (d[i] < min) {
      idx = i;
      min = d[i];
    }
  }

  *m = min;
  return idx;
}

} // namespace acd_gen