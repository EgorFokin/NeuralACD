#include <cost.hpp>
#include <fstream>
#include <hausdorff.hpp>
#include <iostream>
#include <mesh.hpp>

using namespace std;

namespace acd_gen {

void MergeMesh(Mesh &mesh1, Mesh &mesh2, Mesh &merge) {
  merge.vertices.insert(merge.vertices.end(), mesh1.vertices.begin(),
                        mesh1.vertices.end());
  merge.vertices.insert(merge.vertices.end(), mesh2.vertices.begin(),
                        mesh2.vertices.end());
  merge.triangles.insert(merge.triangles.end(), mesh1.triangles.begin(),
                         mesh1.triangles.end());
  int N = mesh1.vertices.size();
  for (int i = 0; i < (int)mesh2.triangles.size(); i++)
    merge.triangles.push_back({mesh2.triangles[i][0] + N,
                               mesh2.triangles[i][1] + N,
                               mesh2.triangles[i][2] + N});
}

double Volume(Vec3D p1, Vec3D p2, Vec3D p3) {
  double v321 = p3[0] * p2[1] * p1[2];
  double v231 = p2[0] * p3[1] * p1[2];
  double v312 = p3[0] * p1[1] * p2[2];
  double v132 = p1[0] * p3[1] * p2[2];
  double v213 = p2[0] * p1[1] * p3[2];
  double v123 = p1[0] * p2[1] * p3[2];
  return (1.0 / 6.0) * (-v321 + v231 + v312 - v132 - v213 + v123);
}
double MeshVolume(Mesh &mesh) {
  double volume = 0;
  for (int i = 0; i < (int)mesh.triangles.size(); i++) {
    int idx0 = mesh.triangles[i][0], idx1 = mesh.triangles[i][1],
        idx2 = mesh.triangles[i][2];
    volume +=
        Volume(mesh.vertices[idx0], mesh.vertices[idx1], mesh.vertices[idx2]);
  }
  return volume;
}

double ComputeRv(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH, double k,
                 double epsilon) {
  double v1, v2, v3;

  v1 = MeshVolume(cvx1);
  v2 = MeshVolume(cvx2);
  v3 = MeshVolume(cvxCH);

  double d = pow(3 * fabs(v1 + v2 - v3) / (4 * Pi), 1.0 / 3) * k;

  return d;
}

double ComputeRv(Mesh &tmesh1, Mesh &tmesh2, double k, double epsilon) {
  double v1, v2;
  v1 = MeshVolume(tmesh1);
  v2 = MeshVolume(tmesh2);

  double d = pow(3 * fabs(v1 - v2) / (4 * Pi), 1.0 / 3) * k;

  return d;
}

double ComputeHb(Mesh &tmesh1, Mesh &tmesh2, unsigned int resolution,
                 unsigned int seed, bool flag) {
  vector<Vec3D> samples1, samples2;
  vector<int> sample_tri_ids1, sample_tri_ids2;

  tmesh1.ExtractPointSet(samples1, sample_tri_ids1, seed, resolution, 1);
  tmesh2.ExtractPointSet(samples2, sample_tri_ids2, seed, resolution, 1);
  // cout << "flag " << flag << endl;
  // if (flag) {
  //   ofstream outFile1("points1.out");
  //   ofstream outFile2("points2.out");
  //   cout << "123\n";

  //   for (auto point : samples1) {
  //     outFile1 << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
  //   }
  //   for (auto point : samples2) {
  //     outFile2 << point[0] << ' ' << point[1] << ' ' << point[2] << '\n';
  //   }

  //   outFile1.close();
  //   outFile2.close();
  // }

  if (!((int)samples1.size() > 0 && (int)samples2.size() > 0))
    return INF;

  double h;
  h = face_hausdorff_distance(tmesh1, samples1, sample_tri_ids1, tmesh2,
                              samples2, sample_tri_ids2);

  return h;
}

double ComputeHb(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH, unsigned int resolution,
                 unsigned int seed) {
  if (cvx1.vertices.size() + cvx2.vertices.size() == cvxCH.vertices.size())
    return 0.0;
  Mesh cvx;
  vector<Vec3D> samples1, samples2;
  vector<int> sample_tri_ids1, sample_tri_ids2;
  MergeMesh(cvx1, cvx2, cvx);
  ExtractPointSet(cvx1, cvx2, samples1, sample_tri_ids1, seed, resolution);
  cvxCH.ExtractPointSet(samples2, sample_tri_ids2, seed, resolution, 1);

  if (!((int)samples1.size() > 0 && (int)samples2.size() > 0))
    return INF;

  double h = face_hausdorff_distance(cvx, samples1, sample_tri_ids1, cvxCH,
                                     samples2, sample_tri_ids2);

  return h;
}

double ComputeHCost(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH, double k,
                    unsigned int resolution, unsigned int seed,
                    double epsilon) {
  double h1 = ComputeRv(cvx1, cvx2, cvxCH, k, epsilon);
  double h2 = ComputeHb(cvx1, cvx2, cvxCH, resolution + 2000, seed);

  return max(h1, h2);
}

double ComputeHCost(Mesh &tmesh1, Mesh &tmesh2, double k,
                    unsigned int resolution, unsigned int seed, double epsilon,
                    bool flag) {
  double h1 = ComputeRv(tmesh1, tmesh2, k, epsilon);
  double h2 = ComputeHb(tmesh1, tmesh2, resolution, seed, flag);

  // cout << "rv: " << h1 << ", hb: " << h2 << endl;
  return max(h1, h2);
}

double MeshDist(Mesh &ch1, Mesh &ch2) {
  vector<Vec3D> XA = ch1.vertices, XB = ch2.vertices;

  int nA = XA.size();

  PointCloud<double> cloudB;
  vec2PointCloud(cloudB, XB);

  typedef KDTreeSingleIndexAdaptor<
      L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>,
      3 /* dim */
      >
      my_kd_tree_t;

  my_kd_tree_t indexB(3 /*dim*/, cloudB,
                      KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
  indexB.buildIndex();

  double minDist = INF;
  for (int i = 0; i < nA; i++) {
    size_t num_results = 1;

    double query_pt[3] = {XA[i][0], XA[i][1], XA[i][2]};

    std::vector<size_t> ret_index(num_results);
    std::vector<double> out_dist_sqr(num_results);

    num_results = indexB.knnSearch(&query_pt[0], num_results, &ret_index[0],
                                   &out_dist_sqr[0]);
    double dist = sqrt(out_dist_sqr[0]);
    minDist = min(minDist, dist);
  }

  return minDist;
}

} // namespace acd_gen