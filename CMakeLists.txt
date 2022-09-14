cmake_minimum_required(VERSION 3.10)

set (CMAKE_CXX_STANDARD 11)

project(FastASR)

set(CMAKE_CXX_FLAGS, "-fPIC")

add_subdirectory("pybind11")
add_subdirectory(lib)
add_subdirectory(examples)
