#include <algorithm>
#include <clip.hpp>
#include <config.hpp>
#include <core.hpp>
#include <cost.hpp>
#include <fstream>
#include <iostream>
#include <jlinkage.hpp>
#include <map>
#include <preprocess.hpp>
#include <process.hpp>
#include <queue>
#include <unordered_map>

using namespace std;

namespace neural_acd {

int32_t find_min_element(const std::vector<double> d, double *const m,
                         const int32_t begin, const int32_t end) {
  int32_t idx = -1;
  double min = (std::numeric_limits<double>::max)();
  for (size_t i = begin; i < size_t(end); ++i) {
    if (d[i] < min) {
      idx = i;
      min = d[i];
    }
  }

  *m = min;
  return idx;
}

void merge_ch(Mesh &ch1, Mesh &ch2, Mesh &ch) {
  Mesh merge;
  merge.vertices.insert(merge.vertices.end(), ch1.vertices.begin(),
                        ch1.vertices.end());
  merge.vertices.insert(merge.vertices.end(), ch2.vertices.begin(),
                        ch2.vertices.end());
  merge.triangles.insert(merge.triangles.end(), ch1.triangles.begin(),
                         ch1.triangles.end());
  for (int i = 0; i < (int)ch2.triangles.size(); i++)
    merge.triangles.push_back({int(ch2.triangles[i][0] + ch1.vertices.size()),
                               int(ch2.triangles[i][1] + ch1.vertices.size()),
                               int(ch2.triangles[i][2] + ch1.vertices.size())});
  merge.compute_ch(ch);
}

void print_cost_mtx(const vector<double> &costMatrix) {
  for (size_t i = 0; i < costMatrix.size(); ++i) {
    if (costMatrix[i] == INF)
      cout << "INF ";
    else
      cout << costMatrix[i] << " ";
  }
  cout << endl;
}

double multimerge_ch(Mesh &m, MeshList &meshs, MeshList &cvxs,
                     double threshold) {
  size_t nConvexHulls = (size_t)cvxs.size();
  double h = 0;

  if (nConvexHulls > 1) {
    int bound = ((((nConvexHulls - 1) * nConvexHulls)) >> 1);
    // Populate the cost matrix
    vector<double> costMatrix;
    costMatrix.resize(bound); // only keeps the top half of the matrix

    size_t p1, p2;
    for (int idx = 0; idx < bound; ++idx) {
      p1 = (int)(sqrt(8 * idx + 1) - 1) >>
           1; // compute nearest triangle number index
      int sum =
          (p1 * (p1 + 1)) >> 1; // compute nearest triangle number from index
      p2 = idx - sum;           // modular arithmetic from triangle number
      p1++;
      double dist = mesh_dist(cvxs[p1], cvxs[p2]);
      if (dist < threshold) {
        Mesh combinedCH;
        merge_ch(cvxs[p1], cvxs[p2], combinedCH);

        costMatrix[idx] = compute_rv(cvxs[p1], cvxs[p2], combinedCH);
      } else {
        costMatrix[idx] = INF;
      }
    }

    size_t costSize = (size_t)cvxs.size();
    while (true) {
      // print_cost_mtx(costMatrix);
      // Search for lowest cost
      double bestCost = INF;
      const int32_t addr = find_min_element(costMatrix, &bestCost, 0,
                                            (int32_t)costMatrix.size());

      // std::cout << "best cost: " << bestCost << " at addr: " << addr
      //           << std::endl;

      if (addr < 0) {
        break;
      }

      if (bestCost > threshold)
        break;

      h = max(h, bestCost);
      const size_t addrI =
          (static_cast<int32_t>(sqrt(1 + (8 * addr))) - 1) >> 1;
      const size_t p1 = addrI + 1;
      const size_t p2 = addr - ((addrI * (addrI + 1)) >> 1);
      // printf("%ld\n", cvxs.size());
      // printf("addr %ld, addrI %ld, p1 %ld, p2 %ld\n", addr, addrI, p1, p2);

      // Make the lowest cost row and column into a new hull
      Mesh cch;
      merge_ch(cvxs[p1], cvxs[p2], cch);
      cvxs[p2] = cch;

      swap(cvxs[p1], cvxs[cvxs.size() - 1]);
      cvxs.pop_back();

      costSize = costSize - 1;

      // Calculate costs versus the new hull
      size_t rowIdx = ((p2 - 1) * p2) >> 1;
      for (size_t i = 0; (i < p2); ++i) {
        double dist = mesh_dist(cvxs[p2], cvxs[i]);
        if (dist < threshold) {
          Mesh combinedCH;
          merge_ch(cvxs[p2], cvxs[i], combinedCH);
          costMatrix[rowIdx++] = compute_rv(cvxs[p2], cvxs[i], combinedCH);
        } else
          costMatrix[rowIdx++] = INF;
      }

      rowIdx += p2;
      for (size_t i = p2 + 1; (i < costSize); ++i) {
        double dist = mesh_dist(cvxs[p2], cvxs[i]);
        if (dist < threshold) {
          Mesh combinedCH;
          merge_ch(cvxs[p2], cvxs[i], combinedCH);
          costMatrix[rowIdx] = compute_rv(cvxs[p2], cvxs[i], combinedCH);
        } else
          costMatrix[rowIdx] = INF;
        rowIdx += i;
      }

      // Move the top column in to replace its space
      const size_t erase_idx = ((costSize - 1) * costSize) >> 1;
      if (p1 < costSize) {
        rowIdx = (addrI * p1) >> 1;
        size_t top_row = erase_idx;
        for (size_t i = 0; i < p1; ++i) {
          if (i != p2) {
            costMatrix[rowIdx] = costMatrix[top_row];
          }
          ++rowIdx;
          ++top_row;
        }

        ++top_row;
        rowIdx += p1;
        for (size_t i = p1 + 1; i < costSize; ++i) {
          costMatrix[rowIdx] = costMatrix[top_row];
          rowIdx += i;
        }
      }
      costMatrix.resize(erase_idx);
    }
  }

  return h;
}

MeshList separate_disjoint_step(Mesh &part) {
  if (part.triangles.empty()) {
    return {};
  }

  map<pair<int, int>, vector<int>> edge_map; // edge -> connected tris

  auto make_edge = [](int a, int b) {
    return std::pair<int, int>{std::min(a, b), std::max(a, b)};
  };

  // Build edge to triangle map
  for (int i = 0; i < part.triangles.size(); ++i) {
    const auto &tri = part.triangles[i];
    pair<int, int> e1 = make_edge(tri[0], tri[1]);
    pair<int, int> e2 = make_edge(tri[1], tri[2]);
    pair<int, int> e3 = make_edge(tri[0], tri[2]);

    edge_map[e1].push_back(i);
    edge_map[e2].push_back(i);
    edge_map[e3].push_back(i);
  }

  unordered_map<int, int> tri_to_part; // triangle index -> part number
  int part_num = 0;

  // Flood fill to find connected components
  for (int i = 0; i < part.triangles.size(); ++i) {
    if (tri_to_part.count(i)) {
      continue;
    }

    queue<int> q;
    q.push(i);
    tri_to_part[i] = part_num;

    while (!q.empty()) {
      int cur = q.front();
      q.pop();

      const auto &tri = part.triangles[cur];
      pair<int, int> edges[3] = {make_edge(tri[0], tri[1]),
                                 make_edge(tri[1], tri[2]),
                                 make_edge(tri[0], tri[2])};

      for (const auto &edge : edges) {
        for (int neighbor : edge_map[edge]) {
          if (neighbor != cur && !tri_to_part.count(neighbor)) {
            tri_to_part[neighbor] = part_num;
            q.push(neighbor);
          }
        }
      }
    }
    part_num++;
  }

  // Create new meshes for each part
  MeshList new_parts(part_num);
  vector<unordered_map<int, int>> vertex_remap(part_num); // global v -> part v

  for (int i = 0; i < part.triangles.size(); ++i) {
    int part_idx = tri_to_part[i];
    Mesh &current_part = new_parts[part_idx];
    const auto &tri = part.triangles[i];

    // Remap vertices
    array<int, 3> new_indices;
    for (int k = 0; k < 3; ++k) {
      int global_v = tri[k];
      auto &remap = vertex_remap[part_idx];

      if (!remap.count(global_v)) {
        remap[global_v] = current_part.vertices.size();
        current_part.vertices.push_back(part.vertices[global_v]);
      }
      new_indices[k] = remap[global_v];
    }

    current_part.triangles.push_back(new_indices);
  }
  return new_parts;
}

void separate_disjoint(MeshList &parts) {
  MeshList new_parts;
  for (auto &part : parts) {
    MeshList res = separate_disjoint_step(part);
    new_parts.insert(new_parts.end(), res.begin(), res.end());
  }
  parts.clear();
  parts.insert(parts.end(), new_parts.begin(), new_parts.end());
}

void write_stats(std::string stats_file, double concavity, int n_parts) {
  ofstream f(stats_file, ios::app);

  f << to_string(concavity) << ";";
  f << to_string(n_parts) << "\n";
  f.close();
}

MeshList process(Mesh mesh, vector<Vec3D> cut_points, std::string stats_file) {
  // cout << cut_points.size() << " cut points provided." << endl;

  mesh.normalize(cut_points); // normalize the mesh and cut points

  MeshList parts;
  if (cut_points.size() != 0) {

    JLinkage jlinkage(config.jlinkage_sigma, config.jlinkage_num_samples,
                      config.jlinkage_threshold,
                      config.jlinkage_outlier_threshold);
    jlinkage.set_points(cut_points);
    vector<Plane> planes = jlinkage.get_best_planes();

    if (planes.empty()) {
      parts.push_back(mesh);
    } else {
      parts = multiclip(mesh, planes);
    }

  } else {
    parts.push_back(mesh);
  }
  separate_disjoint(parts);
  MeshList cvxs;
  for (auto &part : parts) {
    Mesh ch;
    part.compute_ch(ch);
    cvxs.push_back(ch);
    double h = compute_h(part, ch, config.cost_rv_k, config.pcd_res);
    // cout << h << endl;
  }

  // std::cout << "Merge threshold: " << config.merge_threshold << std::endl;
  multimerge_ch(mesh, parts, cvxs, config.merge_threshold);

  Mesh hull;
  int vertex_offset = 0;
  for (auto &cvx : cvxs) {
    hull.vertices.insert(hull.vertices.end(), cvx.vertices.begin(),
                         cvx.vertices.end());
    for (auto &tri : cvx.triangles) {
      hull.triangles.push_back({tri[0] + vertex_offset, tri[1] + vertex_offset,
                                tri[2] + vertex_offset});
    }
    vertex_offset += cvx.vertices.size();
  }
  manifold_preprocess(hull, config.remesh_res, config.remesh_threshold);

  double h = compute_h(mesh, hull, config.cost_rv_k, config.pcd_res);
  cout << "Final concavity: " << h << endl;
  cout << "Number of parts: " << cvxs.size() << endl;

  if (!stats_file.empty())
    write_stats(stats_file, h, cvxs.size());

  // cvxs.push_back(hull);

  if (config.process_output_parts)
    return parts;
  return cvxs;
}

} // namespace neural_acd