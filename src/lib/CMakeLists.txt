
file(GLOB files1 "*.cpp")
file(GLOB files2 "paddlespeech/*.cpp")
file(GLOB files3 "kaldi2/*.cpp")
file(GLOB files4 "paraformer/*.cpp")

set(files ${files1} ${files2} ${files3} ${files4})

# message("${files}")

add_library(fastasr ${files})

if(WIN32)
    if(FASTASR_BUILD_PYTHON_MODULE)
        set(EXTRA_LIBS fftw3f libopenblas webrtcvad)
        target_include_directories(fastasr PUBLIC ${OPENBLAS_INCLUDE_DIR} ${FFTW3F_INCLUDE_DIR})
        target_link_directories(fastasr PUBLIC ${OPENBLAS_LIBRARY_DIR} ${FFTW3F_LIBRARY_DIR})
    else()
        set(EXTRA_LIBS libfftw3f-3 libopenblas webrtcvad)
        if(CMAKE_CL_64)
            target_link_directories(fastasr PUBLIC ${CMAKE_SOURCE_DIR}/win/lib/x64)
        else()
            target_link_directories(fastasr PUBLIC ${CMAKE_SOURCE_DIR}/win/lib/x86)
        endif()
        target_include_directories(fastasr PUBLIC ${CMAKE_SOURCE_DIR}/win/include ${CMAKE_SOURCE_DIR}/win/include/openblas)
    endif()
else()

    set(EXTRA_LIBS fftw3f openblas webrtcvad pthread)
    target_include_directories(fastasr PUBLIC "/usr/local/opt/fftw/include")
    target_link_directories(fastasr PUBLIC "/usr/local/opt/fftw/lib")

    target_include_directories(fastasr PUBLIC "/usr/local/opt/openblas/include")
    target_link_directories(fastasr PUBLIC "/usr/local/opt/openblas/lib")

    target_include_directories(fastasr PUBLIC "/usr/include")
    target_link_directories(fastasr PUBLIC "/usr/lib64")

    target_include_directories(fastasr PUBLIC ${OPENBLAS_INCLUDE_DIR} ${FFTW3F_INCLUDE_DIR})
    target_link_directories(fastasr PUBLIC ${OPENBLAS_LIBRARY_DIR} ${FFTW3F_LIBRARY_DIR})

endif()



include_directories(${CMAKE_SOURCE_DIR}/include)
target_link_libraries(fastasr PUBLIC ${EXTRA_LIBS})



