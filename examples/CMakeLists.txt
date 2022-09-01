

set(EXTRA_LIBS fastasr)
set(EXTRA_INCLUDES ../lib)




add_executable(fastasr_cli fastasr_cli.cpp)

target_link_libraries(fastasr_cli PUBLIC ${EXTRA_LIBS})

include_directories("../include")

target_include_directories(fastasr_cli PUBLIC
    "${PROJECT_BINARY_DIR}"
    ${EXTRA_INCLUDES}
    )














# add_executable(fastasr_stream fastasr_stream.cpp)

# target_link_libraries(fastasr_stream PUBLIC ${EXTRA_LIBS})

# target_include_directories(fastasr_stream PUBLIC
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
