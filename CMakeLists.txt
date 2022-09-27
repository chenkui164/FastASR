cmake_minimum_required(VERSION 3.10)


project(FastASR)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(FASTASR_BUILD_PYTHON_MODULE "build python module, using FastASR in Python" OFF)

if(FASTASR_BUILD_PYTHON_MODULE)
    add_subdirectory("./third_party/pybind11")
endif()

add_subdirectory(lib)
add_subdirectory(examples)
