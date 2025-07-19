#pragma once

#include <mesh.hpp>
#include <nanoflann.hpp>

namespace acd_gen {

struct PointCloud {
  std::vector<Vec3D> pts;

  // Must return the number of data points
  inline size_t kdtree_get_point_count() const { return pts.size(); }

  // Returns the dim-th component of the idx-th point in the class
  inline double kdtree_get_pt(const size_t idx, const size_t dim) const {
    if (dim == 0)
      return pts[idx][0];
    else if (dim == 1)
      return pts[idx][1];
    else
      return pts[idx][2];
  }
  // Optional bounding-box computation
  template <class BBOX> bool kdtree_get_bbox(BBOX &) const { return false; }
};
using KDTree = nanoflann::KDTreeSingleIndexAdaptor<
    nanoflann::L2_Simple_Adaptor<double, PointCloud>, PointCloud,
    3 // dimensionality
    >;
std::vector<std::vector<int>>
euclidean_clustering_nanoflann(const std::vector<Vec3D> &points,
                               double eps = 0.05);

} // namespace acd_gen