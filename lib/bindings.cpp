#include <clip.hpp>
#include <config.hpp>
#include <core.hpp>
#include <cost.hpp>
#include <dbscan.hpp>
#include <generate.hpp>
#include <iostream>
#include <jlinkage.hpp>
#include <preprocess.hpp>
#include <process.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 3>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 4>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::array<int, 3>>);
PYBIND11_MAKE_OPAQUE(
    std::vector<std::vector<std::array<double, 3>>>); // cut verts
PYBIND11_MAKE_OPAQUE(std::vector<double>);
PYBIND11_MAKE_OPAQUE(std::vector<int>);

PYBIND11_MODULE(lib_neural_acd, m) {
  py::bind_vector<std::vector<std::array<double, 3>>>(
      m, "VecArray3d"); // 3D vector array
  py::bind_vector<std::vector<std::array<int, 3>>>(
      m, "VecArray3i"); // triangle array
  py::bind_vector<std::vector<std::array<double, 4>>>(
      m, "VecArray4d"); // plane array
  py::bind_vector<std::vector<double>>(m, "VecDouble");
  py::bind_vector<std::vector<int>>(m, "VecInt"); // sample triangle ids
  py::bind_vector<std::vector<std::vector<std::array<double, 3>>>>(
      m, "VecVecArray3d"); // cut verts

  py::class_<neural_acd::Mesh>(m, "Mesh")
      .def_readwrite("vertices", &neural_acd::Mesh::vertices)
      .def_readwrite("triangles", &neural_acd::Mesh::triangles)
      .def_readwrite("cut_verts", &neural_acd::Mesh::cut_verts)
      .def(py::init<>())
      .def("extract_point_set",
           static_cast<void (neural_acd::Mesh::*)(
               std::vector<std::array<double, 3>> &, std::vector<int> &,
               size_t)>(&neural_acd::Mesh::extract_point_set),
           py::arg("samples"), py::arg("sample_tri_ids"),
           py::arg("resolution") = 10000);

  py::bind_vector<neural_acd::MeshList>(m, "MeshList");

  py::class_<neural_acd::Config>(m, "Config")
      .def(py::init<>())
      .def_readwrite("generation_cuboid_width_min",
                     &neural_acd::Config::generation_cuboid_width_min)
      .def_readwrite("generation_cuboid_width_max",
                     &neural_acd::Config::generation_cuboid_width_max)
      .def_readwrite("generation_sphere_radius_min",
                     &neural_acd::Config::generation_sphere_radius_min)
      .def_readwrite("generation_sphere_radius_max",
                     &neural_acd::Config::generation_sphere_radius_max)
      .def_readwrite("generation_icosphere_subdivs",
                     &neural_acd::Config::generation_icosphere_subdivs)
      .def_readwrite("pcd_res", &neural_acd::Config::pcd_res)
      .def_readwrite("remesh_res", &neural_acd::Config::remesh_res)
      .def_readwrite("remesh_threshold", &neural_acd::Config::remesh_threshold)
      .def_readwrite("cost_rv_k", &neural_acd::Config::cost_rv_k)
      .def_readwrite("merge_threshold", &neural_acd::Config::merge_threshold)
      .def_readwrite("dbscan_eps", &neural_acd::Config::dbscan_eps)
      .def_readwrite("dbscan_min_pts", &neural_acd::Config::dbscan_min_pts)
      .def_readwrite("dbscan_outlier_threshold",
                     &neural_acd::Config::dbscan_outlier_threshold)
      .def_readwrite("jlinkage_sigma", &neural_acd::Config::jlinkage_sigma)
      .def_readwrite("jlinkage_num_samples",
                     &neural_acd::Config::jlinkage_num_samples)
      .def_readwrite("jlinkage_threshold",
                     &neural_acd::Config::jlinkage_threshold)
      .def_readwrite("jlinkage_outlier_threshold",
                     &neural_acd::Config::jlinkage_outlier_threshold)
      .def_readwrite("refinement_iterations",
                     &neural_acd::Config::refinement_iterations)
      .def_readwrite("process_output_parts",
                     &neural_acd::Config::process_output_parts);

  m.def("make_vecarray3i", [](py::array_t<int> input) {
    auto buf = input.request();
    std::vector<std::array<int, 3>> result;

    int X = buf.shape[0];
    int *ptr = (int *)buf.ptr;

    for (size_t idx = 0; idx < X; idx++) {
      std::array<int, 3> arr;
      arr[0] = ptr[idx * 3];
      arr[1] = ptr[idx * 3 + 1];
      arr[2] = ptr[idx * 3 + 2];
      result.push_back(arr);
    }

    return result;
  });

  m.attr("config") =
      py::cast(&neural_acd::config, py::return_value_policy::reference);

  m.def("generate_cuboid_structure", &neural_acd::generate_cuboid_structure,
        py::arg("obj_num"));
  m.def("generate_sphere_structure", &neural_acd::generate_sphere_structure,
        py::arg("obj_num"));
  m.def("set_seed", &neural_acd::set_seed, py::arg("seed"));
  m.def("clip", &neural_acd::clip, py::arg("mesh"), py::arg("plane_args"));
  m.def("multiclip", &neural_acd::multiclip, py::arg("mesh"),
        py::arg("planes"));
  m.def("process", &neural_acd::process, py::arg("mesh"), py::arg("cut_points"),
        py::arg("stats_file") = "");
  m.def("preprocess", &neural_acd::manifold_preprocess, py::arg("mesh"),
        py::arg("scale"), py::arg("level_set"));
  m.def(
      "get_best_planes",
      [](std::vector<neural_acd::Vec3D> cut_points) {
        std::vector<neural_acd::Plane> planes =
            neural_acd::get_best_planes(cut_points);
        std::vector<std::array<double, 4>> result;
        for (const auto &plane : planes) {
          result.push_back({plane.a, plane.b, plane.c, plane.d});
        }
        return result;
      },
      py::arg("points"));
  m.def(
      "dbscan",
      [](const std::vector<neural_acd::Vec3D> &data, double eps, int min_pts) {
        return neural_acd::dbscan(data, eps, min_pts);
      },
      py::arg("data"), py::arg("eps"), py::arg("min_pts"));

  m.def(
      "test",
      [](neural_acd::Mesh m1, neural_acd::Mesh m2) {
        double h1 = neural_acd::compute_h(m1, m2, 0.03, 3000, 42);
        std::cout << "h1: " << h1 << std::endl;
      },
      py::arg("mesh1"), py::arg("mesh2"));
}
