
pybind11_add_module(_fastasr PyFastASR.cpp)
target_include_directories(_fastasr PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(WIN32)
    target_link_directories(_fastasr PUBLIC ${OPENBLAS_LIBRARY_DIR} ${FFTW3F_LIBRARY_DIR})
    install(FILES  ${OPENBLAS_BIN_DIR}/libopenblas.dll DESTINATION .)
endif()


target_link_libraries(_fastasr PUBLIC fastasr)
install(TARGETS _fastasr DESTINATION .)
