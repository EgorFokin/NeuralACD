cmake_minimum_required(VERSION 3.14)
project(acd_gen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)



include(FetchContent)
FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
  GIT_TAG        v2.11.1  # or the version you prefer
)
FetchContent_MakeAvailable(pybind11)




file(GLOB ACD_GEN_SOURCES "src/*.cpp") #all files under src

add_library(acd_gen STATIC ${ACD_GEN_SOURCES})

#Add 3rd party cdt lib
target_include_directories(acd_gen PRIVATE 3rd/CDT/CDT/include)

target_include_directories(acd_gen PUBLIC ${CMAKE_SOURCE_DIR}/include)

pybind11_add_module(lib_acd_gen bindings.cpp)

target_link_libraries(lib_acd_gen PRIVATE acd_gen)

