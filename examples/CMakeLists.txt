

if(WIN32)
    if(CMAKE_CL_64)
        link_directories( ${CMAKE_SOURCE_DIR}/win/lib/x64 )
    else()
        link_directories( ${CMAKE_SOURCE_DIR}/win/lib/x86 )
    endif()
endif()

set(EXTRA_LIBS fastasr)


include_directories(${CMAKE_SOURCE_DIR}/include)

add_executable(paddlespeech_cli paddlespeech_cli.cpp)
target_link_libraries(paddlespeech_cli PUBLIC ${EXTRA_LIBS})


add_executable(k2_rnnt2_cli k2_rnnt2_cli.cpp)
target_link_libraries(k2_rnnt2_cli PUBLIC ${EXTRA_LIBS})

add_executable(paddlespeech_stream paddlespeech_stream.cpp)
target_link_libraries(paddlespeech_stream PUBLIC ${EXTRA_LIBS})


configure_file(k2_rnnt2_cli.py k2_rnnt2_cli.py COPYONLY)
configure_file(paddlespeech_cli.py paddlespeech_cli.py COPYONLY)
configure_file(paddlespeech_stream.py paddlespeech_stream.py COPYONLY)

# target_include_directories(paddlespeech_cli PUBLIC
#     "${PROJECT_BINARY_DIR}"
#     ${EXTRA_INCLUDES}
#     )




# target_include_directories(k2_rnnt2_cli PUBLIC
#     "${PROJECT_BINARY_DIR}"
#     ${EXTRA_INCLUDES}
#     )





# target_include_directories(paddlespeech_stream PUBLIC
#     "${PROJECT_BINARY_DIR}"
#     ${EXTRA_INCLUDES}
#     )





# add_executable(fastasr_cli_c fastasr_cli_c.c)

# target_link_libraries(fastasr_cli_c PUBLIC ${EXTRA_LIBS})

# target_include_directories(fastasr_cli_c PUBLIC
#     "${PROJECT_BINARY_DIR}"
#     ${EXTRA_INCLUDES}
#     )


# add_executable(fastasr_stream_c fastasr_stream_c.c)

# target_link_libraries(fastasr_stream_c PUBLIC ${EXTRA_LIBS})

# target_include_directories(fastasr_stream_c PUBLIC
#     "${PROJECT_BINARY_DIR}"
#     ${EXTRA_INCLUDES}
#     )
