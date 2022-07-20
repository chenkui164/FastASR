
FILE(GLOB SRCFILES *.cpp)
# message("${SRCFILES}")

add_library(fastasr ${SRCFILES})

set(EXTRA_LIBS fftw3f openblas)
target_link_libraries(fastasr PUBLIC ${EXTRA_LIBS})
