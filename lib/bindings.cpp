#include <clip.hpp>
#include <generate.hpp>
#include <mesh.hpp>
#include <preprocess.hpp>
#include <process.hpp>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 3>>);
PYBIND11_MAKE_OPAQUE(std::vector<std::array<double, 4>>);

PYBIND11_MODULE(lib_acd_gen, m) {
  py::bind_vector<std::vector<std::array<double, 3>>>(
      m, "VecArray3d"); // 3D vector array
  py::bind_vector<std::vector<std::array<double, 4>>>(
      m, "VecArray4d");                                 // plane array
  py::bind_vector<std::vector<double>>(m, "VecDouble"); // cut verts
  py::class_<acd_gen::Mesh>(m, "Mesh")
      .def_readwrite("vertices", &acd_gen::Mesh::vertices)
      .def_readwrite("triangles", &acd_gen::Mesh::triangles)
      .def_readwrite("cut_verts", &acd_gen::Mesh::cut_verts);
  py::bind_vector<acd_gen::MeshList>(m, "MeshList");

  m.def("generate_cuboid_structure", &acd_gen::generate_cuboid_structure,
        py::arg("obj_num"));
  m.def("generate_sphere_structure", &acd_gen::generate_sphere_structure,
        py::arg("obj_num"), py::arg("min_radius") = 0.1,
        py::arg("max_radius") = 0.5);
  m.def("test", &acd_gen::test);
  m.def("set_seed", &acd_gen::set_seed, py::arg("seed"));
  m.def("clip", &acd_gen::clip, py::arg("mesh"), py::arg("plane_args"));
  m.def("multiclip", &acd_gen::multiclip, py::arg("mesh"), py::arg("planes"));
  m.def("process", &acd_gen::process, py::arg("mesh"), py::arg("cut_points"));
  m.def("preprocess", &acd_gen::ManifoldPreprocess, py::arg("mesh"),
        py::arg("scale"), py::arg("level_set"));
}
