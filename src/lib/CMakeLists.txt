
file(GLOB files1 "*.cpp")
file(GLOB files2 "paddlespeech/*.cpp")
file(GLOB files3 "kaldi2/*.cpp")

set(files ${files1} ${files2} ${files3})

# message("${files}")

add_library(fastasr ${files})

if(WIN32)
    if(CMAKE_CL_64)
        link_directories(${CMAKE_SOURCE_DIR}/win/lib/x64)
    else()
        link_directories(${CMAKE_SOURCE_DIR}/win/lib/x86)
    endif()

    include_directories(${CMAKE_SOURCE_DIR}/win/include ${CMAKE_SOURCE_DIR}/win/include/openblas )
    set(EXTRA_LIBS libfftw3f-3 libopenblas webrtcvad)
else()

    set(EXTRA_LIBS fftw3f openblas webrtcvad pthread)
    target_include_directories(fastasr PUBLIC "/usr/local/opt/fftw/include")
    target_link_directories(fastasr PUBLIC "/usr/local/opt/fftw/lib")

    target_include_directories(fastasr PUBLIC "/usr/local/opt/openblas/include")
    target_link_directories(fastasr PUBLIC "/usr/local/opt/openblas/lib")

    target_include_directories(fastasr PUBLIC "/usr/include/openblas")
    target_include_directories(fastasr PUBLIC "/usr/include")
    target_link_directories(fastasr PUBLIC "/usr/lib64")

endif()



include_directories(${CMAKE_SOURCE_DIR}/include)
target_link_libraries(fastasr PUBLIC ${EXTRA_LIBS})



