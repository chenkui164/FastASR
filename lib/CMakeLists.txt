

file(GLOB files1 "*.cpp")
file(GLOB files2 "paddlespeech/*.cpp")
file(GLOB files3 "kaldi2/*.cpp")
set(files ${files1} ${files2} ${files3})
message("${files}")

add_library(fastasr ${files})

set(EXTRA_LIBS fftw3f openblas)

# target_include_directories(fastasr PUBLIC "paddlespeech")
include_directories("../include")
target_link_libraries(fastasr PUBLIC ${EXTRA_LIBS})

target_include_directories(fastasr PUBLIC "/usr/local/opt/fftw/include")
target_link_directories(fastasr PUBLIC "/usr/local/opt/fftw/lib")

target_include_directories(fastasr PUBLIC "/usr/local/opt/openblas/include")
target_link_directories(fastasr PUBLIC "/usr/local/opt/openblas/lib")
