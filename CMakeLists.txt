cmake_minimum_required(VERSION 3.0...3.26)

project(index_test
    LANGUAGES CXX)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")

find_package(hnswlib REQUIRED)
add_library(hnswlib INTERFACE)

include_directories(src)

set(CMAKE_CXX_STANDARD 11)

# Build Config
if (CMAKE_BUILD_TYPE MATCHES "Debug")
  SET( CMAKE_CXX_FLAGS  "-O0 -g -std=c++11 -DHAVE_CXX0X -fpic -ftree-vectorize" )
else()
  SET( CMAKE_CXX_FLAGS  "-Ofast -lrt -std=c++11 -DHAVE_CXX0X -march=native -fpic -w -ftree-vectorize -ftree-vectorizer-verbose=0" )
endif()

# Targets
add_executable(timestamp_graph_test test/timestamp_graph_test.cpp)
target_link_libraries(timestamp_graph_test hnswlib)

add_executable(neighbor_tree_test test/neighbor_tree_test.cpp)
target_link_libraries(neighbor_tree_test hnswlib)