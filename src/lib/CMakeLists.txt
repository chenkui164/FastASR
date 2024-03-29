project(fastasr)

file(GLOB files1 "*.cpp")
file(GLOB files2 "paddlespeech/*.cpp")
file(GLOB files3 "kaldi2/*.cpp")
file(GLOB files4 "paraformer/*.cpp")

set(files ${files1} ${files2} ${files3} ${files4})

# message("${files}")
add_library(${PROJECT_NAME} ${files})

if(WIN32)
    if(FASTASR_BUILD_PYTHON_MODULE)
        set(EXTRA_LIBS fftw3f libopenblas webrtcvad)
        target_include_directories(${PROJECT_NAME} PUBLIC ${OPENBLAS_INCLUDE_DIR} ${FFTW3F_INCLUDE_DIR})
        target_link_directories(${PROJECT_NAME} PUBLIC ${OPENBLAS_LIBRARY_DIR} ${FFTW3F_LIBRARY_DIR})
    else()
        set(EXTRA_LIBS libfftw3f-3 libopenblas webrtcvad)

        if(CMAKE_CL_64)
            target_link_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../win/lib/x64)
        else()
            target_link_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../win/lib/x86)
        endif()

        target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../win/include ${CMAKE_CURRENT_LIST_DIR}/../../win/include/openblas)
    endif()
else()
    set(EXTRA_LIBS fftw3f openblas webrtcvad pthread)
    target_include_directories(${PROJECT_NAME} PUBLIC "/usr/local/opt/fftw/include")
    target_link_directories(${PROJECT_NAME} PUBLIC "/usr/local/opt/fftw/lib")

    target_include_directories(${PROJECT_NAME} PUBLIC "/usr/local/opt/openblas/include")
    target_link_directories(${PROJECT_NAME} PUBLIC "/usr/local/opt/openblas/lib")

    target_include_directories(${PROJECT_NAME} PUBLIC "/usr/include")
    target_link_directories(${PROJECT_NAME} PUBLIC "/usr/lib64")

    target_include_directories(${PROJECT_NAME} PUBLIC ${OPENBLAS_INCLUDE_DIR} ${FFTW3F_INCLUDE_DIR})
    target_link_directories(${PROJECT_NAME} PUBLIC ${OPENBLAS_LIBRARY_DIR} ${FFTW3F_LIBRARY_DIR})
endif()

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_LIST_DIR}/../../include)
target_link_libraries(${PROJECT_NAME} PUBLIC ${EXTRA_LIBS})
