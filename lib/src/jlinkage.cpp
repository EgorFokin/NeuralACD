#include <Eigen/Dense>
#include <algorithm>
#include <bitset>
#include <cmath>
#include <core.hpp>
#include <fstream>
#include <iostream>
#include <jlinkage.hpp>
#include <limits>
#include <random>
#include <stdexcept>

using namespace std;

namespace neural_acd {
JLinkage::JLinkage(double sigma_, int num_samples_, double threshold_,
                   int outlier_threshold_)
    : sigma(sigma_), num_samples(num_samples_), threshold(threshold_),
      outlier_threshold(outlier_threshold_) {
  if (num_samples > SAMPLE_LIMIT) {
    throw std::invalid_argument("Jlinkage: num_samples is too large");
  }
}

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
  uniform_int_distribution<> dis(0, points.size() - 1);

  i1 = dis(random_engine);
  discrete_distribution<> dis2(sample_probs[i1].begin(),
                               sample_probs[i1].end());
  i2 = dis2(random_engine);

  vector<double> prob3(points.size(), 0.0);
  for (int i = 0; i < points.size(); ++i) {
    prob3[i] = sample_probs[i1][i] * sample_probs[i2][i];
  }
  double sum = accumulate(prob3.begin(), prob3.end(), 0.0);
  for (auto &p : prob3)
    p /= sum;
  discrete_distribution<> dis3(prob3.begin(), prob3.end());
  i3 = dis3(random_engine);

  // cout << "Sampled triplet: " << i1 << ", " << i2 << ", " << i3 << endl;
  // cout << "Probability of this point:" << sample_probs[i1][i2] << endl;
  // cout << "Probability of this point: " << sample_probs[i1][i3] << endl;
}

void JLinkage::calculate_preference_sets() {
  int N = points.size();
  preference_set = vector<bitset<SAMPLE_LIMIT>>(N);

  for (int i = 0; i < num_samples; ++i) {
    int i1, i2, i3;
    sample_triplet(i1, i2, i3);
    Vec3D p1 = points[i1], p2 = points[i2], p3 = points[i3];

    Vec3D v1 = {p2[0] - p1[0], p2[1] - p1[1], p2[2] - p1[2]};
    Vec3D v2 = {p3[0] - p1[0], p3[1] - p1[1], p3[2] - p1[2]};
    Vec3D n = cross_product(v1, v2);
    double n_norm = vector_length(n);
    if (n_norm == 0)
      continue;
    for (double &c : n)
      c /= n_norm;
    double d = -dot(n, p1);

    for (int j = 0; j < N; ++j) {
      double dist = abs(dot(points[j], n) + d);
      if (dist < threshold) {
        preference_set[j][i] = true;
      }
    }
  }
}

double JLinkage::jaccard_distance(const std::bitset<SAMPLE_LIMIT> &a,
                                  const std::bitset<SAMPLE_LIMIT> &b) {
  auto intersection = (a & b).count();
  auto union_count = (a | b).count();
  return 1.0 - static_cast<double>(intersection) / (union_count + 1e-8);
}

void print_dist_matrix(const vector<double> &dist_matrix, int N) {
  int idx = 0;
  for (int i = 1; i < N; ++i) {
    for (int j = 0; j < i; ++j) {
      cout << dist_matrix[idx] << " ";
      idx++;
    }
    cout << endl;
  }
}

vector<Plane> JLinkage::get_best_planes() {
  int N = points.size();
  // N = 5;
  vector<double> dist_matrix(N * (N - 1) >> 1, 0.0);

  int i1, i2;
  for (int i = 0; i < dist_matrix.size(); ++i) {
    i1 = (int)(sqrt(8 * i + 1) - 1) >> 1;
    int sum = (i1 * (i1 + 1)) >> 1;
    i2 = i - sum;
    i1++;
    dist_matrix[i] = jaccard_distance(preference_set[i1], preference_set[i2]);
  }
  // print_dist_matrix(dist_matrix, N);

  vector<vector<int>> clusters(N);
  for (int i = 0; i < N; ++i)
    clusters[i] = {i};

  LoadingBar loading_bar("Jlinkage", clusters.size());
  while (true) {
    loading_bar.step();
    double min_val = numeric_limits<double>::infinity();
    int mini = -1, minj = -1;

    for (int i = 0; i < dist_matrix.size(); ++i)
      if (dist_matrix[i] < min_val) {
        min_val = dist_matrix[i];
        mini = (int)(sqrt(8 * i + 1) - 1) >> 1;
        int sum = (mini * (mini + 1)) >> 1;
        minj = i - sum;
        mini++;
      }

    if (min_val >= 1 || mini == -1 || minj == -1)
      break;
    for (size_t k = 0; k < preference_set[mini].size(); ++k)
      preference_set[mini][k] =
          preference_set[mini][k] && preference_set[minj][k];

    clusters[mini].insert(clusters[mini].end(), clusters[minj].begin(),
                          clusters[minj].end());
    preference_set.erase(preference_set.begin() + minj);
    clusters.erase(clusters.begin() + minj);

    // delete column j
    int idx = dist_matrix.size() - (N - 1) + minj; // last row, column j
    for (int i = 0; i < N - minj - 1; ++i) {
      dist_matrix.erase(dist_matrix.begin() + idx);
      idx -= (N - 2 - i);
    }

    // delete row j
    idx = (minj * (minj - 1)) >> 1; // row j, first column
    for (int i = idx + minj - 1; i > idx - 1; --i) {
      dist_matrix.erase(dist_matrix.begin() + i);
    }

    if (mini > minj)
      mini--; // adjust mini if it was after minj

    N--;
    // update column i
    idx = dist_matrix.size() - (N - 1) + mini; // last row, column i
    for (int i = 0; i < N - mini - 1; ++i) {
      i1 = (int)(sqrt(8 * idx + 1) - 1) >> 1;
      int sum = (i1 * (i1 + 1)) >> 1;
      i2 = idx - sum;
      i1++;
      dist_matrix[idx] =
          jaccard_distance(preference_set[i1], preference_set[i2]);
      idx -= (N - 2 - i);
    }

    // update row i
    idx = (mini * (mini - 1)) >> 1; // row i, first column
    for (int i = idx; i < idx + mini; ++i) {
      i1 = (int)(sqrt(8 * i + 1) - 1) >> 1;
      int sum = (i1 * (i1 + 1)) >> 1;
      i2 = i - sum;
      i1++;
      dist_matrix[i] = jaccard_distance(preference_set[i1], preference_set[i2]);
    }

    // print_dist_matrix(dist_matrix, N);
  }
  loading_bar.finish();
  vector<Plane> planes = cluster_planes(clusters);
  cout << "Found " << planes.size() << " planes" << endl;

  return planes;
}

vector<Plane> JLinkage::cluster_planes(vector<vector<int>> &clusters) {
  vector<Plane> planes;
  for (auto &cluster : clusters) {
    if (cluster.size() < outlier_threshold)
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
    double d = -dot(normal, centroid);
    Plane plane(a, b, c, d);
    planes.push_back(plane);
  }
  return planes;
}
} // namespace neural_acd