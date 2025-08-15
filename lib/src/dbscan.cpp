#include "dbscan.hpp"

#include <cstddef>
#include <nanoflann.hpp>

#include <core.hpp>
#include <type_traits>
#include <vector>

namespace neural_acd {

template <typename Point> struct adaptor {
  const std::span<const Point> &points;
  adaptor(const std::span<const Point> &points) : points(points) {}

  /// CRTP helper method
  // inline const Derived& derived() const { return obj; }

  // Must return the number of data points
  inline std::size_t kdtree_get_point_count() const { return points.size(); }

  // Returns the dim'th component of the idx'th point in the class:
  // Since this is inlined and the "dim" argument is typically an immediate
  // value, the
  //  "if/else's" are actually solved at compile time.
  inline double kdtree_get_pt(const std::size_t idx,
                              const std::size_t dim) const {
    return points[idx][dim];
  }

  // Optional bounding-box computation: return false to default to a standard
  // bbox computation loop.
  //   Return true if the BBOX was already computed by the class and returned in
  //   "bb" so it can be avoided to redo it again. Look at bb.size() to find out
  //   the expected dimensionality (e.g. 2 or 3 for point clouds)
  template <class BBOX> bool kdtree_get_bbox(BBOX & /*bb*/) const {
    return false;
  }

  auto const *elem_ptr(const std::size_t idx) const { return &points[idx][0]; }
};

auto sort_clusters(std::vector<std::vector<size_t>> &clusters) {
  for (auto &cluster : clusters) {
    std::sort(cluster.begin(), cluster.end());
  }
}

template <int n_cols, typename Adaptor>
auto dbscan(const Adaptor &adapt, double eps, int min_pts) {
  eps *= eps;
  using namespace nanoflann;
  using my_kd_tree_t =
      KDTreeSingleIndexAdaptor<L2_Simple_Adaptor<double, decltype(adapt)>,
                               decltype(adapt), n_cols>;

  auto index = my_kd_tree_t(n_cols, adapt, KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();

  const auto n_points = adapt.kdtree_get_point_count();
  auto visited = std::vector<bool>(n_points);
  auto clusters = std::vector<std::vector<size_t>>();
  auto matches = std::vector<std::pair<size_t, double>>();
  auto sub_matches = std::vector<std::pair<size_t, double>>();

  for (size_t i = 0; i < n_points; i++) {
    if (visited[i])
      continue;

    index.radiusSearch(adapt.elem_ptr(i), eps, matches,
                       SearchParams(32, 0.f, false));
    if (matches.size() < static_cast<size_t>(min_pts))
      continue;
    visited[i] = true;

    auto cluster = std::vector({i});

    while (matches.empty() == false) {
      auto nb_idx = matches.back().first;
      matches.pop_back();
      if (visited[nb_idx])
        continue;
      visited[nb_idx] = true;

      index.radiusSearch(adapt.elem_ptr(nb_idx), eps, sub_matches,
                         SearchParams(32, 0.f, false));

      if (sub_matches.size() >= static_cast<size_t>(min_pts)) {
        std::copy(sub_matches.begin(), sub_matches.end(),
                  std::back_inserter(matches));
      }
      cluster.push_back(nb_idx);
    }
    clusters.emplace_back(std::move(cluster));
  }
  sort_clusters(clusters);
  return clusters;
}

std::vector<std::vector<size_t>> dbscan(const std::span<const Vec3D> &data,
                                        double eps, int min_pts) {
  const auto adapt = adaptor<Vec3D>(data);

  return dbscan<3>(adapt, eps, min_pts);
}
} // namespace neural_acd