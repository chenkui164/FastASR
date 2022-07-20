
FILE(GLOB SRCFILES *.cpp)
# message("${SRCFILES}")

add_library(fastasr ${SRCFILES})

set(EXTRA_LIBS fftw3f openblas)

target_link_libraries(fastasr PUBLIC ${EXTRA_LIBS})

target_include_directories(fastasr PUBLIC "/usr/local/opt/fftw/include")
target_link_directories(fastasr PUBLIC "/usr/local/opt/fftw/lib")

target_include_directories(fastasr PUBLIC "/usr/local/opt/openblas/include")
target_link_directories(fastasr PUBLIC "/usr/local/opt/openblas/lib")
