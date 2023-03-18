project(webrtcvad)

if(WIN32)
    add_definitions(-DWEBRTC_WIN)
else()
    add_definitions(-DWEBRTC_POSIX)
endif()

file(GLOB_RECURSE files "*.c" "rtc_base/checks.cc")

message("${files}")

add_library(${PROJECT_NAME} OBJECT ${files})

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}/..
    ${CMAKE_CURRENT_LIST_DIR}/common_audio/signal_processing/include
)
