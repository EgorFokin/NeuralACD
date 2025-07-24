#include <clip.hpp>
#include <config.hpp>
#include <core.hpp>
#include <generate.hpp>
#include <preprocess.hpp>
#include <process.hpp>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 3>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 4>>);

PYBIND11_MODULE(lib_neural_acd, m) {
  py::bind_vector<std::vector<std::array<double, 3>>>(
      m, "VecArray3d"); // 3D vector array
  py::bind_vector<std::vector<std::array<int, 3>>>(
      m, "VecArray3i"); // triangle array
  py::bind_vector<std::vector<std::array<double, 4>>>(
      m, "VecArray4d");                                 // plane array
  py::bind_vector<std::vector<double>>(m, "VecDouble"); // cut verts

  py::class_<neural_acd::Mesh>(m, "Mesh")
      .def_readwrite("vertices", &neural_acd::Mesh::vertices)
      .def_readwrite("triangles", &neural_acd::Mesh::triangles)
      .def_readwrite("cut_verts", &neural_acd::Mesh::cut_verts)
      .def(py::init<>());

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
      .def_readwrite("jlinkage_sigma", &neural_acd::Config::jlinkage_sigma)
      .def_readwrite("jlinkage_num_samples",
                     &neural_acd::Config::jlinkage_num_samples)
      .def_readwrite("jlinkage_threshold",
                     &neural_acd::Config::jlinkage_threshold)
      .def_readwrite("jlinkage_outlier_threshold",
                     &neural_acd::Config::jlinkage_outlier_threshold)
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
}
