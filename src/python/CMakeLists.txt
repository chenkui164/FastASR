
include_directories(${CMAKE_SOURCE_DIR}/include)
pybind11_add_module(_fastasr PyFastASR.cpp)
target_link_libraries(_fastasr PUBLIC fastasr)
install(TARGETS _fastasr DESTINATION .)
