#pragma once
#include <core.hpp>
#include <utility>
#include <vector>

using namespace std;

namespace neural_acd {

namespace detail {
using BoolVec = vector<bool>;
using BoolMat = vector<BoolVec>;
} // namespace detail

class JLinkage {
public:
  JLinkage(double sigma_ = 1, int num_samples_ = 10000, double threshold_ = 0.1,
           int outlier_threshold_ = 10);

  void set_points(const vector<Vec3D> &pts);
  vector<Plane> get_best_planes();

private:
  double sigma;
  int num_samples;
  double threshold;
  int outlier_threshold;
  vector<Vec3D> points;
  vector<vector<double>> sample_probs;
  detail::BoolMat preference_set;
  void calculate_distances();
  void sample_triplet(int &i1, int &i2, int &i3);
  void calculate_preference_sets();
  double jaccard_distance(const detail::BoolVec &a, const detail::BoolVec &b);
  vector<Plane> cluster_planes(vector<vector<int>> &clusters);
};

} // namespace neural_acd