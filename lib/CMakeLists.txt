

file(GLOB files1 "*.cpp")
file(GLOB files2 "paddlespeech/*.cpp")
file(GLOB files3 "kaldi2/*.cpp")

# list(REMOVE_ITEM files1 "/home/ck/work/FastASR/lib/PyFastASR.cpp")
list(FILTER files1 EXCLUDE REGEX ".*PyFastASR.cpp")
set(files ${files1} ${files2} ${files3})

message("${files}")

add_library(fastasr SHARED ${files})

if(WIN32)

	if(CMAKE_CL_64)

	link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../win/lib/x64 )
	else()
	link_directories( ${CMAKE_CURRENT_SOURCE_DIR}/../win/lib/x86 )
	endif()

	include_directories(../win/include ../win/include/openblas )
	set(EXTRA_LIBS libfftw3f-3 libopenblas)
else()
	set(EXTRA_LIBS fftw3f openblas)
endif()



# target_include_directories(fastasr PUBLIC "paddlespeech")
include_directories("../include")
target_link_libraries(fastasr PUBLIC ${EXTRA_LIBS})

target_include_directories(fastasr PUBLIC "/usr/local/opt/fftw/include")
target_link_directories(fastasr PUBLIC "/usr/local/opt/fftw/lib")

target_include_directories(fastasr PUBLIC "/usr/local/opt/openblas/include")
target_link_directories(fastasr PUBLIC "/usr/local/opt/openblas/lib")

pybind11_add_module(PyFastASR PyFastASR.cpp)
target_link_libraries(PyFastASR PUBLIC fastasr)
