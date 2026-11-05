cmake_minimum_required(VERSION 3.22...4.2)
project(issue120_probe LANGUAGES CXX)

set(TT_METAL_SOURCE_ROOT "" CACHE PATH "Exact tt-metal v0.75.0 source tree")
set(TT_METAL_SOURCE_REVISION "" CACHE STRING "Peeled commit for the v0.75.0 source tree")
set(TT_METAL_LIBRARY "" CACHE FILEPATH "libtt_metal.so extracted from the pinned image")
set(TT_METAL_EXTRA_INCLUDE_DIRS "" CACHE STRING "Semicolon-separated third-party/generated include roots")
set(TT_METAL_EXTRA_LIBRARY_DIRS "" CACHE STRING "Semicolon-separated pinned-image dependency library roots")

if(NOT IS_DIRECTORY "${TT_METAL_SOURCE_ROOT}/tt_metal/api")
    message(FATAL_ERROR "TT_METAL_SOURCE_ROOT must be the exact v0.75.0 source tree")
endif()
if(NOT TT_METAL_SOURCE_REVISION STREQUAL "d9a68815f5fcf08a5bfbffb6f1f811823fba8edd")
    message(FATAL_ERROR "TT_METAL_SOURCE_REVISION must identify the pinned v0.75.0 tag")
endif()
if(NOT EXISTS "${TT_METAL_LIBRARY}")
    message(FATAL_ERROR "TT_METAL_LIBRARY must be the staged libtt_metal.so from the pinned image")
endif()
file(SHA256 "${TT_METAL_LIBRARY}" TT_METAL_LIBRARY_SHA256)
if(NOT TT_METAL_LIBRARY_SHA256 STREQUAL "daa26f51aa60299d914d7fb91fbf055394faea321b2b876a4d189a3407f15141")
    message(FATAL_ERROR "TT_METAL_LIBRARY is not the audited library from the pinned image")
endif()

add_executable(issue120_probe issue120_probe.cpp)
target_compile_features(issue120_probe PRIVATE cxx_std_20)
target_compile_definitions(issue120_probe PRIVATE
    ISSUE120_SOURCE_DIR="enodia/tt/bench/issue120"
    SPDLOG_FMT_EXTERNAL
    FMT_HEADER_ONLY
)
target_include_directories(issue120_probe PRIVATE
    "${TT_METAL_SOURCE_ROOT}/tt_metal/api"
    "${TT_METAL_SOURCE_ROOT}/tt_metal/hostdevcommon"
    "${TT_METAL_SOURCE_ROOT}/tt_metal/hw/inc"
    "${TT_METAL_SOURCE_ROOT}/tt_stl"
    ${TT_METAL_EXTRA_INCLUDE_DIRS}
)
target_link_directories(issue120_probe PRIVATE ${TT_METAL_EXTRA_LIBRARY_DIRS})
get_filename_component(TT_METAL_LIBRARY_DIR "${TT_METAL_LIBRARY}" DIRECTORY)
if(NOT EXISTS "${TT_METAL_LIBRARY_DIR}/libtt_stl.so")
    message(FATAL_ERROR "Stage libtt_stl.so beside the pinned-image libtt_metal.so")
endif()
target_link_libraries(issue120_probe PRIVATE
    "${TT_METAL_LIBRARY}" "${TT_METAL_LIBRARY_DIR}/libtt_stl.so"
)
foreach(lib_dir IN LISTS TT_METAL_EXTRA_LIBRARY_DIRS)
    target_link_options(issue120_probe PRIVATE "LINKER:-rpath-link,${lib_dir}")
endforeach()
set_target_properties(issue120_probe PROPERTIES
    BUILD_RPATH "$ORIGIN;/opt/venv/lib/python3.12/site-packages/ttnn/build/lib;/opt/venv/lib/python3.12/site-packages/ttnn.libs"
    INSTALL_RPATH "$ORIGIN;/opt/venv/lib/python3.12/site-packages/ttnn/build/lib;/opt/venv/lib/python3.12/site-packages/ttnn.libs"
    BUILD_RPATH_USE_ORIGIN TRUE
)
add_custom_command(TARGET issue120_probe POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TT_METAL_LIBRARY}" "$<TARGET_FILE_DIR:issue120_probe>/libtt_metal.so"
)
