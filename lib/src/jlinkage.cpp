#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <jlinkage.hpp>
#include <limits>
#include <random>

using namespace std;

namespace acd_gen {
JLinkage::JLinkage(double sigma_, int num_samples_, double threshold_)
    : sigma(sigma_), num_samples(num_samples_), threshold(threshold_) {}

void JLinkage::set_points(const vector<Vec3D> &pts) {
  points = pts;
  calculate_distances();
  calculate_preference_sets();
}

void JLinkage::calculate_distances() {
  int N = points.size();
  vector<vector<double>> D(N, vector<double>(N, 0.0));
  sample_probs = vector<vector<double>>(N, vector<double>(N, 0.0));

  for (int i = 0; i < N; ++i) {
    double row_sum = 0.0;
    for (int j = 0; j < N; ++j) {
      if (i == j) {
        D[i][j] = numeric_limits<double>::infinity();
      } else {
        D[i][j] = vector_length(points[i] - points[j]) *
                  vector_length(points[i] - points[j]);
      }
      sample_probs[i][j] = exp(-D[i][j] / (sigma * sigma));
      row_sum += sample_probs[i][j];
    }
    for (int j = 0; j < N; ++j) {
      sample_probs[i][j] /= row_sum;
    }
  }
}

void JLinkage::sample_triplet(int &i1, int &i2, int &i3) {
  static random_device rd;
  static mt19937 gen(rd());
  uniform_int_distribution<> dis(0, points.size() - 1);

  i1 = dis(gen);
  discrete_distribution<> dis2(sample_probs[i1].begin(),
                               sample_probs[i1].end());
  i2 = dis2(gen);

  vector<double> prob3(points.size(), 0.0);
  for (int i = 0; i < points.size(); ++i) {
    prob3[i] = sample_probs[i1][i] * sample_probs[i2][i];
  }
  double sum = accumulate(prob3.begin(), prob3.end(), 0.0);
  for (auto &p : prob3)
    p /= sum;
  discrete_distribution<> dis3(prob3.begin(), prob3.end());
  i3 = dis3(gen);
}

void JLinkage::calculate_preference_sets() {
  int N = points.size();
  preference_set = BoolMat(N, BoolVec(num_samples, false));

  for (int i = 0; i < num_samples; ++i) {
    int i1, i2, i3;
    sample_triplet(i1, i2, i3);
    Vec3D p1 = points[i1], p2 = points[i2], p3 = points[i3];

    Vec3D v1 = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
    Vec3D v2 = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};
    Vec3D n = CrossProduct(v1, v2);
    double n_norm = vector_length(n);
    if (n_norm == 0)
      continue;
    for (double &c : n)
      c /= n_norm;
    double d = -DotProduct(n, p1);

    for (int j = 0; j < N; ++j) {
      double dist = abs(DotProduct(points[j], n) + d);
      if (dist < threshold) {
        preference_set[j][i] = true;
      }
    }
  }
}

double JLinkage::jaccard_distance(const BoolVec &a, const BoolVec &b) {
  int intersection = 0, union_count = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] || b[i])
      union_count++;
    if (a[i] && b[i])
      intersection++;
  }
  return 1.0 - (double)intersection / (union_count + 1e-8);
}

vector<Plane> JLinkage::get_best_planes() {
  int N = preference_set.size();
  vector<vector<double>> dist_matrix(N, vector<double>(N, 0.0));
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      dist_matrix[i][j] =
          (i == j) ? numeric_limits<double>::infinity()
                   : jaccard_distance(preference_set[i], preference_set[j]);

  vector<vector<int>> clusters(N);
  for (int i = 0; i < N; ++i)
    clusters[i] = {i};

  while (true) {
    double min_val = numeric_limits<double>::infinity();
    int mini = -1, minj = -1;

    for (int i = 0; i < dist_matrix.size(); ++i)
      for (int j = i + 1; j < dist_matrix[i].size(); ++j)
        if (dist_matrix[i][j] < min_val) {
          min_val = dist_matrix[i][j];
          mini = i;
          minj = j;
        }

    if (min_val >= 1 || mini == -1 || minj == -1)
      break;
    cout << clusters.size() << " " << min_val << endl;

    for (size_t k = 0; k < preference_set[mini].size(); ++k)
      preference_set[mini][k] =
          preference_set[mini][k] && preference_set[minj][k];

    clusters[mini].insert(clusters[mini].end(), clusters[minj].begin(),
                          clusters[minj].end());
    preference_set.erase(preference_set.begin() + minj);
    clusters.erase(clusters.begin() + minj);

    for (int i = 0; i < dist_matrix.size(); ++i) {
      dist_matrix[i].erase(dist_matrix[i].begin() + minj);
    }
    dist_matrix.erase(dist_matrix.begin() + minj);

    for (int k = 0; k < dist_matrix.size(); ++k) {
      if (k == mini || k == minj)
        continue;
      dist_matrix[mini][k] = dist_matrix[k][mini] =
          jaccard_distance(preference_set[mini], preference_set[k]);
    }
  }

  vector<Plane> planes = cluster_planes(clusters);
  cout << planes.size() << " planes" << endl;

  // ofstream outFile1("points1.out");
  // ofstream outFile2("points2.out");

  // for (auto idx : clusters[0]) {
  //   outFile1 << points[idx][0] << ' ' << points[idx][1] << ' ' <<
  //   points[idx][2]
  //            << '\n';
  // }
  // for (auto idx : clusters[1]) {
  //   outFile2 << points[idx][0] << ' ' << points[idx][1] << ' ' <<
  //   points[idx][2]
  //            << '\n';
  // }

  // outFile1.close();
  // outFile2.close();

  return planes;
}

vector<Plane> JLinkage::cluster_planes(vector<vector<int>> &clusters) {
  vector<Plane> planes;
  for (auto &cluster : clusters) {
    if (cluster.size() < 70)
      continue;

    Vec3D centroid = {0.0, 0.0, 0.0};
    for (int idx : cluster)
      for (int j = 0; j < 3; ++j)
        centroid[j] += points[idx][j];
    for (int j = 0; j < 3; ++j)
      centroid[j] /= cluster.size();

    Eigen::MatrixXd centered(cluster.size(), 3);
    for (int i = 0; i < cluster.size(); ++i) {
      centered(i, 0) = points[cluster[i]][0] - centroid[0];
      centered(i, 1) = points[cluster[i]][1] - centroid[1];
      centered(i, 2) = points[cluster[i]][2] - centroid[2];
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centered, Eigen::ComputeThinU |
                                                        Eigen::ComputeThinV);
    Eigen::MatrixXd V = svd.matrixV();

    Vec3D normal = {V(0, 2), V(1, 2), V(2, 2)};

    double a = normal[0];
    double b = normal[1];
    double c = normal[2];
    double d = -DotProduct(normal, centroid);
    Plane plane(a, b, c, d);
    planes.push_back(plane);
  }
  return planes;
}
} // namespace acd_gen