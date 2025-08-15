#pragma once

namespace neural_acd {
class Config {
public:
  float generation_cuboid_width_min;
  float generation_cuboid_width_max;
  float generation_sphere_radius_min;
  float generation_sphere_radius_max;
  int generation_icosphere_subdivs;

  int pcd_res;
  float remesh_res;
  float remesh_threshold;

  double cost_rv_k;

  double merge_threshold;

  double dbscan_eps;
  int dbscan_min_pts;
  int dbscan_outlier_threshold;

  double jlinkage_sigma;
  int jlinkage_num_samples;
  double jlinkage_threshold;
  int jlinkage_outlier_threshold;

  int refinement_iterations;

  bool process_output_parts;

  Config() {
    generation_cuboid_width_min = 0.1;
    generation_cuboid_width_max = 0.5;
    generation_sphere_radius_min = 0.1;
    generation_sphere_radius_max = 0.25;
    generation_icosphere_subdivs = 3;

    pcd_res = 3000;

    remesh_res = 50.0f;
    remesh_threshold = 0.05f;

    cost_rv_k = 0.03;

    merge_threshold = 0.005;

    dbscan_eps = 0.07;
    dbscan_min_pts = 10;
    dbscan_outlier_threshold = 20;

    jlinkage_sigma = 1.0;
    jlinkage_num_samples = 10000;
    jlinkage_threshold = 0.1;
    jlinkage_outlier_threshold = 10;

    refinement_iterations = 10;

    process_output_parts = false;
  }
};

inline Config config;

} // namespace neural_acd