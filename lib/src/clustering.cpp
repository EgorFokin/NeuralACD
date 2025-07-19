#include <clustering.hpp>
#include <mesh.hpp>
#include <nanoflann.hpp>
#include <queue>
#include <vector>

namespace acd_gen {

std::vector<std::vector<int>>
euclidean_clustering_nanoflann(const std::vector<Vec3D> &points, double eps) {
  PointCloud cloud{points};
  KDTree index(3, cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
  index.buildIndex();

  int n = points.size();
  std::vector<bool> visited(n, false);
  std::vector<std::vector<int>> clusters;

  nanoflann::SearchParams params;

  for (int i = 0; i < n; ++i) {
    if (visited[i])
      continue;

    std::queue<int> q;
    std::vector<int> cluster;

    q.push(i);
    visited[i] = true;

    while (!q.empty()) {
      int idx = q.front();
      q.pop();
      cluster.push_back(idx);

      std::vector<std::pair<size_t, double>> ret_matches;
      std::vector<double> query_pt = {points[idx][0], points[idx][1],
                                      points[idx][2]};

      index.radiusSearch(&query_pt[0], eps * eps, ret_matches, params);

      for (auto &match : ret_matches) {
        size_t neighbor_idx = match.first;
        if (!visited[neighbor_idx]) {
          visited[neighbor_idx] = true;
          q.push(neighbor_idx);
        }
      }
    }

    clusters.push_back(cluster);
  }

  return clusters;
}

} // namespace acd_gen