#pragma once
#include <mesh.hpp>
#include <utility>
#include <vector>

using namespace std;

namespace acd_gen {

using BoolVec = vector<bool>;
using BoolMat = vector<BoolVec>;

class JLinkage {
public:
  double sigma;
  int num_samples;
  double threshold;
  vector<Vec3D> points;
  vector<vector<double>> sample_probs;
  BoolMat preference_set;

  JLinkage(double sigma_ = 1, int num_samples_ = 10000,
           double threshold_ = 0.1);

  void set_points(const vector<Vec3D> &pts);
  void calculate_distances();
  void sample_triplet(int &i1, int &i2, int &i3);
  void calculate_preference_sets();
  double jaccard_distance(const BoolVec &a, const BoolVec &b);
  vector<Plane> cluster_planes(vector<vector<int>> &clusters);
  vector<Plane> get_best_planes();
};

} // namespace acd_gen