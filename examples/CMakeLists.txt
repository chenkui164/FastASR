add_executable(example_fastasr main.cpp)


set(EXTRA_LIBS fastasr)
set(EXTRA_INCLUDES ../lib)
target_link_libraries(example_fastasr PUBLIC ${EXTRA_LIBS})

target_include_directories(example_fastasr PUBLIC
    "${PROJECT_BINARY_DIR}"
    ${EXTRA_INCLUDES}
    )
