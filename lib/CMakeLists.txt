cmake_minimum_required(VERSION 3.14)
project(acd_gen)

set(CMAKE_CXX_STANDARD 17)
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



file(GLOB ACD_GEN_SOURCES "src/*.cpp") #all files under src

add_library(acd_gen STATIC ${ACD_GEN_SOURCES})



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

target_include_directories(acd_gen PRIVATE
    3rd/CDT/CDT/include
)

target_link_libraries(acd_gen PRIVATE
    quickhull
    btConvexHull
    openvdb_static
    Eigen3::Eigen
)

include(boost)
include(openvdb)

target_include_directories(acd_gen PUBLIC ${CMAKE_SOURCE_DIR}/include)

pybind11_add_module(lib_acd_gen bindings.cpp)

#target_link_options(lib_acd_gen PRIVATE -Wl,--no-undefined)

target_link_libraries(lib_acd_gen PRIVATE acd_gen)

