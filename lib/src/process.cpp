#include <algorithm>
#include <clip.hpp>
#include <clustering.hpp>
#include <cost.hpp>
#include <fstream>
#include <iostream>
#include <jlinkage.hpp>
#include <map>
#include <mesh.hpp>
#include <preprocess.hpp>
#include <process.hpp>
#include <queue>

using namespace std;

namespace acd_gen {

void MergeCH(Mesh &ch1, Mesh &ch2, Mesh &ch) {
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
  merge.ComputeCH(ch);
}

void print_cost_matrix(vector<double> matrix) {
  for (int r = 0; r < matrix.size(); r++) {
    // for (int c = 0; c <= r.size(); c++) {
    cout << matrix[r] << ' ';
    // }
  }
  cout << endl;
}

double MergeConvexHulls(Mesh &m, MeshList &meshs, MeshList &cvxs,
                        double epsilon, double threshold) {
  size_t nConvexHulls = (size_t)cvxs.size();
  double h = 0;

  if (nConvexHulls > 1) {
    int bound = ((((nConvexHulls - 1) * nConvexHulls)) >> 1);
    // Populate the cost matrix
    vector<double> costMatrix, precostMatrix;
    costMatrix.resize(bound);    // only keeps the top half of the matrix
    precostMatrix.resize(bound); // only keeps the top half of the matrix

    size_t p1, p2;
    for (int idx = 0; idx < bound; ++idx) {
      p1 = (int)(sqrt(8 * idx + 1) - 1) >>
           1; // compute nearest triangle number index
      int sum =
          (p1 * (p1 + 1)) >> 1; // compute nearest triangle number from index
      p2 = idx - sum;           // modular arithmetic from triangle number
      p1++;
      double dist = MeshDist(cvxs[p1], cvxs[p2]);
      if (dist < threshold) {
        Mesh combinedCH;
        MergeCH(cvxs[p1], cvxs[p2], combinedCH);

        costMatrix[idx] =
            ComputeRv(cvxs[p1], cvxs[p2], combinedCH, 0.03, 0.0001);
        precostMatrix[idx] =
            max(ComputeHCost(meshs[p1], cvxs[p1], 0.03, 3000, 42),
                ComputeHCost(meshs[p2], cvxs[p2], 0.03, 3000, 42));
      } else {
        costMatrix[idx] = INF;
      }
    }
    // print_cost_matrix(costMatrix);

    size_t costSize = (size_t)cvxs.size();

    while (true) {
      // Search for lowest cost
      double bestCost = INF;
      const int32_t addr = FindMinimumElement(costMatrix, &bestCost, 0,
                                              (int32_t)costMatrix.size());

      if (addr < 0) {
        break;
      }

      // if dose not set max nConvexHull, stop the merging when bestCost is
      // larger than the threshold
      if (bestCost > threshold)
        break;
      if (bestCost - precostMatrix[addr] > 0.03) {
        costMatrix[addr] = INF;
        continue;
      }

      h = max(h, bestCost);
      const size_t addrI =
          (static_cast<int32_t>(sqrt(1 + (8 * addr))) - 1) >> 1;
      const size_t p1 = addrI + 1;
      const size_t p2 = addr - ((addrI * (addrI + 1)) >> 1);
      // printf("addr %ld, addrI %ld, p1 %ld, p2 %ld\n", addr, addrI, p1, p2);

      // Make the lowest cost row and column into a new hull
      Mesh cch;
      MergeCH(cvxs[p1], cvxs[p2], cch);
      cvxs[p2] = cch;

      swap(cvxs[p1], cvxs[cvxs.size() - 1]);
      cvxs.pop_back();

      costSize = costSize - 1;

      // Calculate costs versus the new hull
      size_t rowIdx = ((p2 - 1) * p2) >> 1;
      for (size_t i = 0; (i < p2); ++i) {
        double dist = MeshDist(cvxs[p2], cvxs[i]);
        if (dist < threshold) {
          Mesh combinedCH;
          MergeCH(cvxs[p2], cvxs[i], combinedCH);
          costMatrix[rowIdx] =
              ComputeRv(cvxs[p2], cvxs[i], combinedCH, 0.03, 0.0001);
          precostMatrix[rowIdx++] =
              max(precostMatrix[p2] + bestCost, precostMatrix[i]);
        } else
          costMatrix[rowIdx++] = INF;
      }

      rowIdx += p2;
      for (size_t i = p2 + 1; (i < costSize); ++i) {
        double dist = MeshDist(cvxs[p2], cvxs[i]);
        if (dist < threshold) {
          Mesh combinedCH;
          MergeCH(cvxs[p2], cvxs[i], combinedCH);
          costMatrix[rowIdx] =
              ComputeRv(cvxs[p2], cvxs[i], combinedCH, 0.03, 0.0001);
          precostMatrix[rowIdx] =
              max(precostMatrix[p2] + bestCost, precostMatrix[i]);
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
            precostMatrix[rowIdx] = precostMatrix[top_row];
          }
          ++rowIdx;
          ++top_row;
        }

        ++top_row;
        rowIdx += p1;
        for (size_t i = p1 + 1; i < costSize; ++i) {
          costMatrix[rowIdx] = costMatrix[top_row];
          precostMatrix[rowIdx] = precostMatrix[top_row++];
          rowIdx += i;
        }
      }
      costMatrix.resize(erase_idx);
      precostMatrix.resize(erase_idx);
      // print_cost_matrix(costMatrix);
    }
  }

  return h;
}

MeshList SeparateDisjoint_step(Mesh &part) {
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

  map<int, int> tri_to_part; // triangle index -> part number
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

void SeparateDisjoint(MeshList &parts) {
  MeshList new_parts;
  for (auto &part : parts) {
    MeshList res = SeparateDisjoint_step(part);
    new_parts.insert(new_parts.end(), res.begin(), res.end());
  }
  parts.clear();
  parts.insert(parts.end(), new_parts.begin(), new_parts.end());
}

void write_stats(double concavity, int n_parts) {
  ofstream stats_file("stats.txt", ios::app);

  stats_file << to_string(concavity) << ";";
  stats_file << to_string(n_parts) << "\n";
  stats_file.close();
}

MeshList process(Mesh mesh, vector<Vec3D> cut_points) {
  cout << cut_points.size() << " cut points provided." << endl;

  MeshList parts;
  if (cut_points.size() != 0) {

    JLinkage jlinkage;
    jlinkage.set_points(cut_points);
    vector<Plane> planes = jlinkage.get_best_planes();

    parts = multiclip(mesh, planes);
  } else {
    parts.push_back(mesh);
  }
  SeparateDisjoint(parts);
  MeshList cvxs;
  for (auto &part : parts) {
    Mesh ch;
    part.ComputeCH(ch);
    cvxs.push_back(ch);
    double h = ComputeHCost(part, ch, 0.03, 3000, 42);
    cout << h << endl;
  }

  MergeConvexHulls(mesh, parts, cvxs, 0.0001, 0.002);

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
  ManifoldPreprocess(hull, 50.0f, 0.05f);

  double h = ComputeHCost(mesh, hull, 0.03, 3000, 42);
  cout << "Final concavity: " << h << endl;

  write_stats(h, cvxs.size());

  // cvxs.push_back(hull);
  return cvxs;
}

} // namespace acd_gen