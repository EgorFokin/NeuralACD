cmake_minimum_required(VERSION 3.14)
project(neural_acd)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-stringop-overflow")
set(OPENVDB_CORE_SHARED OFF)
set(TBB_TEST OFF)


list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")

include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1  # or the version you prefer
)
FetchContent_MakeAvailable(pybind11)

FetchContent_Declare(
  eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0  # or latest stable
)
FetchContent_MakeAvailable(eigen)



file(GLOB neural_acd_SOURCES "src/*.cpp") #all files under src

add_library(neural_acd STATIC ${neural_acd_SOURCES})



#add 3rd party libs
add_library(quickhull STATIC
    3rd/quickhull/QuickHull.cpp
)
target_include_directories(quickhull PUBLIC
    3rd/quickhull
)

add_library(btConvexHull STATIC
    3rd/btConvexHull/btConvexHullComputer.cpp
    3rd/btConvexHull/btAlignedAllocator.cpp
)
target_include_directories(btConvexHull PUBLIC
    3rd/btConvexHull
)

target_include_directories(neural_acd PRIVATE
    3rd/CDT/CDT/include
)

target_link_libraries(neural_acd PRIVATE
    quickhull
    btConvexHull
    openvdb_static
    Eigen3::Eigen
)

include(boost)
include(openvdb)

target_include_directories(neural_acd PUBLIC ${CMAKE_SOURCE_DIR}/include)

pybind11_add_module(lib_neural_acd bindings.cpp)

#target_link_options(lib_neural_acd PRIVATE -Wl,--no-undefined)

target_link_libraries(lib_neural_acd PRIVATE neural_acd)

