cmake_minimum_required(VERSION 3.24)

project(gpf-algorithm VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(cmake/CPM.cmake)

set(CGAL_DO_NOT_WARN_ABOUT_CMAKE_BUILD_TYPE ON)
find_package(CGAL REQUIRED)

# CPMAddPackage("gh:liukaizheng/mesh#main")
CPMAddPackage(
    NAME mesh
    SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../mesh
)
CPMAddPackage("gh:liukaizheng/predicates#main")
CPMFindPackage(
    NAME Eigen3
    VERSION 5.0.1
    GITLAB_REPOSITORY libeigen/eigen
    GIT_TAG 5.0.1
)

add_library(gpf-algorithm INTERFACE)
add_library(gpf::algorithm ALIAS gpf-algorithm)

target_compile_features(gpf-algorithm INTERFACE cxx_std_23)
target_link_libraries(gpf-algorithm INTERFACE
    gpf::mesh
    predicates
    Eigen3::Eigen
    CGAL::CGAL
)
target_include_directories(gpf-algorithm INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set(GPF_ALGORITHM_IS_TOP_LEVEL OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
  set(GPF_ALGORITHM_IS_TOP_LEVEL ON)
endif()

option(GPF_ALGORITHM_BUILD_TESTS "Build tests" ${GPF_ALGORITHM_IS_TOP_LEVEL})

if(GPF_ALGORITHM_BUILD_TESTS)
  enable_testing()
  add_executable(
    gpf_algorithm_tests
    tests/main.cpp
    tests/test_triangulation.cpp
    tests/test_mesh_property.cpp
    tests/test_mesh_edge_collapse.cpp
    tests/test_mesh_flood_fill.cpp
    tests/test_collapse_short_edges_2d.cpp
    tests/test_orthtree.cpp
    tests/test_project_polylines_on_mesh.cpp
  )
  target_link_libraries(gpf_algorithm_tests PRIVATE gpf::algorithm)
  add_test(NAME gpf_algorithm_tests COMMAND gpf_algorithm_tests)
endif()
