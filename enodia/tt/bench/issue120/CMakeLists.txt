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

# The coordinate type is ABI-sensitive: newer UMD moves xy_pair's namespace.
# v0.75.0 pins UMD at bc77ac2f7382ffeb9fa5430055f0854d53979489.
set(UMD_API "${TT_METAL_SOURCE_ROOT}/tt_metal/third_party/umd/device/api")
if(NOT EXISTS "${UMD_API}/umd/device/types/xy_pair.hpp")
    message(FATAL_ERROR "Populate the v0.75.0 pinned UMD submodule before building")
endif()
file(SHA256 "${UMD_API}/umd/device/types/xy_pair.hpp" UMD_COORDINATE_SHA256)
if(NOT UMD_COORDINATE_SHA256 STREQUAL "80ad8fc434b617c13b2b58a3c5a95b740fc3d87053715177ab9c352324608aba")
    message(FATAL_ERROR "UMD coordinate header does not match the v0.75.0 dependency")
endif()

if(NOT EXISTS "${TT_METAL_SOURCE_ROOT}/tt_metal/impl/context/metal_context.hpp"
   OR NOT EXISTS "${TT_METAL_SOURCE_ROOT}/tt_metal/llrt/tt_cluster.hpp"
   OR NOT EXISTS "${TT_METAL_SOURCE_ROOT}/tt_metal/distributed/mesh_workload_impl.hpp")
    message(FATAL_ERROR "Stage the pinned internal headers for targeted L1 observation")
endif()

# Generate the exact tag's HAL interfaces without executing any device code.
find_package(Python3 COMPONENTS Interpreter REQUIRED)
set(HAL_CODEGEN "${TT_METAL_SOURCE_ROOT}/tt_metal/llrt/hal/codegen/codegen.py")
set(HAL_OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/generated/hal/generated")
set(HAL_NAMES dev_msgs fabric_telemetry realtime_profiler_msgs)
set(HAL_INPUTS dev_msgs.h fabric_telemetry_msgs.h realtime_profiler_msgs.h)
set(HAL_NAMESPACES tt::tt_metal::dev_msgs tt::tt_fabric::fabric_telemetry tt::tt_metal::realtime_profiler_msgs)
set(HAL_HEADERS)
foreach(index RANGE 0 2)
    list(GET HAL_NAMES ${index} name)
    list(GET HAL_INPUTS ${index} input)
    list(GET HAL_NAMESPACES ${index} interface_namespace)
    set(input_file "${TT_METAL_SOURCE_ROOT}/tt_metal/hw/inc/hostdev/${input}")
    add_custom_command(
        OUTPUT "${HAL_OUTPUT}/${name}.hpp" "${HAL_OUTPUT}/${name}_impl.hpp"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${HAL_OUTPUT}"
        COMMAND "${Python3_EXECUTABLE}" "${HAL_CODEGEN}"
            --driver_ns=::tt::tt_metal::hal_structs
            --driver_include_path=llrt/struct_view_driver.hpp
            "--interface_ns=${interface_namespace}"
            "${input_file}" "${HAL_OUTPUT}/${name}.hpp" "${HAL_OUTPUT}/${name}_impl.hpp"
        DEPENDS "${HAL_CODEGEN}" "${input_file}"
        VERBATIM
    )
    list(APPEND HAL_HEADERS "${HAL_OUTPUT}/${name}.hpp" "${HAL_OUTPUT}/${name}_impl.hpp")
endforeach()
add_custom_target(issue120_hal_headers DEPENDS ${HAL_HEADERS})

add_executable(issue120_probe issue120_probe.cpp q4_frame_probe.cpp)
add_dependencies(issue120_probe issue120_hal_headers)
target_compile_features(issue120_probe PRIVATE cxx_std_20)
target_compile_definitions(issue120_probe PRIVATE
    ISSUE120_SOURCE_DIR="enodia/tt/bench/issue120"
    SPDLOG_FMT_EXTERNAL
    FMT_HEADER_ONLY
)
target_include_directories(issue120_probe PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/generated"
    "${TT_METAL_SOURCE_ROOT}"
    "${TT_METAL_SOURCE_ROOT}/tt_metal"
    "${TT_METAL_SOURCE_ROOT}/tt_metal/impl"
    "${TT_METAL_SOURCE_ROOT}/tt_metal/api/tt-metalium"
    "${TT_METAL_SOURCE_ROOT}/tt_metal/api"
    "${TT_METAL_SOURCE_ROOT}/tt_metal/hostdevcommon/api"
    "${TT_METAL_SOURCE_ROOT}/tt_metal/hostdevcommon"
    "${TT_METAL_SOURCE_ROOT}/tt_metal/hw/inc"
    "${TT_METAL_SOURCE_ROOT}/tt_stl"
    "${UMD_API}"
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
