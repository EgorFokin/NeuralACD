#pragma once
#include <bitset>
#include <core.hpp>
#include <utility>
#include <vector>

using namespace std;

namespace neural_acd {

constexpr int SAMPLE_LIMIT = 20000; // Maximum number of points

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
  vector<bitset<SAMPLE_LIMIT>> preference_set;
  void calculate_distances();
  void sample_triplet(int &i1, int &i2, int &i3);
  void calculate_preference_sets();
  double jaccard_distance(const bitset<SAMPLE_LIMIT> &a,
                          const bitset<SAMPLE_LIMIT> &b);
  vector<Plane> cluster_planes(vector<vector<int>> &clusters);
};

} // namespace neural_acd