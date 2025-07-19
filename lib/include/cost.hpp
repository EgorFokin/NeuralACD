#pragma once

#include <mesh.hpp>

namespace acd_gen {
constexpr double Pi = 3.14159265;
double ComputeRv(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH, double k, double epsilon);
double ComputeHb(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH, unsigned int resolution,
                 unsigned int seed);
double ComputeHCost(Mesh &cvx1, Mesh &cvx2, Mesh &cvxCH, double k,
                    unsigned int resolution, unsigned int seed,
                    double epsilon = 0.0001);
double ComputeHCost(Mesh &tmesh1, Mesh &tmesh2, double k,
                    unsigned int resolution, unsigned int seed = 1235,
                    double epsilon = 0.0001, bool flag = false);
double MeshDist(Mesh &tmesh1, Mesh &tmesh2);

} // namespace acd_gen