

file(GLOB files1 "*.cpp")
file(GLOB files2 "paddlespeech/*.cpp")
file(GLOB files3 "kaldi2/*.cpp")

# list(REMOVE_ITEM files1 "/home/ck/work/FastASR/lib/PyFastASR.cpp")
list(FILTER files1 EXCLUDE REGEX ".*PyFastASR.cpp")
set(files ${files1} ${files2} ${files3})

message("${files}")

add_library(fastasr ${files})

if(WIN32)

    if(CMAKE_CL_64)
        link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../win/lib/x64 )
    else()
        link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../win/lib/x86 )
    endif()

    include_directories(../win/include ../win/include/openblas )
    set(EXTRA_LIBS libfftw3f-3 libopenblas webrtcvad)
else()
    set(EXTRA_LIBS fftw3f openblas webrtcvad pthread)
endif()



# target_include_directories(fastasr PUBLIC "paddlespeech")
include_directories("../include")
target_link_libraries(fastasr PUBLIC ${EXTRA_LIBS})

target_include_directories(fastasr PUBLIC "/usr/local/opt/fftw/include")
target_link_directories(fastasr PUBLIC "/usr/local/opt/fftw/lib")

target_include_directories(fastasr PUBLIC "/usr/local/opt/openblas/include")
target_link_directories(fastasr PUBLIC "/usr/local/opt/openblas/lib")


if(FASTASR_BUILD_PYTHON_MODULE)
    pybind11_add_module(PyFastASR PyFastASR.cpp)
    target_link_libraries(PyFastASR PUBLIC fastasr)
endif()
