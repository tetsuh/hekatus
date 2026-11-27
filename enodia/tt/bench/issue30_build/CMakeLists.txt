cmake_minimum_required(VERSION 3.22)
project(issue30_official_tests LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_FLAGS "" CACHE STRING "No inherited compiler flags" FORCE)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SOURCE /source)
set(DEPS /deps)
set(RUNTIME /opt/venv/lib/python3.12/site-packages/ttnn)
set(LIB ${RUNTIME}/build/lib)
file(SHA256 ${LIB}/libtt_metal.so METAL_SHA)
if(NOT METAL_SHA STREQUAL "daa26f51aa60299d914d7fb91fbf055394faea321b2b876a4d189a3407f15141")
    message(FATAL_ERROR "Pinned runtime library mismatch")
endif()
set(BUILD_GMOCK ON CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
add_subdirectory(${DEPS}/googletest ${CMAKE_BINARY_DIR}/gtest EXCLUDE_FROM_ALL)
set(HAL_OUT ${CMAKE_BINARY_DIR}/generated/hal/generated)
file(MAKE_DIRECTORY ${HAL_OUT})
foreach(name dev_msgs fabric_telemetry realtime_profiler_msgs)
    if(name STREQUAL dev_msgs)
        set(input dev_msgs.h)
        set(ns tt::tt_metal::dev_msgs)
    elseif(name STREQUAL fabric_telemetry)
        set(input fabric_telemetry_msgs.h)
        set(ns tt::tt_fabric::fabric_telemetry)
    else()
        set(input realtime_profiler_msgs.h)
        set(ns tt::tt_metal::realtime_profiler_msgs)
    endif()
    execute_process(COMMAND /opt/venv/bin/python ${SOURCE}/tt_metal/llrt/hal/codegen/codegen.py
        --driver_ns=::tt::tt_metal::hal_structs --driver_include_path=llrt/struct_view_driver.hpp
        --interface_ns=${ns} ${SOURCE}/tt_metal/hw/inc/hostdev/${input}
        ${HAL_OUT}/${name}.hpp ${HAL_OUT}/${name}_impl.hpp COMMAND_ERROR_IS_FATAL ANY)
endforeach()
set(INCLUDES ${CMAKE_BINARY_DIR}/generated ${SOURCE} ${SOURCE}/tests ${SOURCE}/tt_metal
    ${SOURCE}/tt_metal/impl ${SOURCE}/tt_metal/api ${SOURCE}/tt_metal/api/tt-metalium
    ${SOURCE}/tt_metal/hostdevcommon/api ${SOURCE}/tt_metal/hostdevcommon
    ${SOURCE}/tt_metal/hw/inc ${SOURCE}/tt_stl
    ${SOURCE}/tt_metal/third_party/umd/device/api
    ${SOURCE}/tests/tt_metal/tt_fabric/common
    ${DEPS}/fmt/include ${DEPS}/json/include ${DEPS}/enchantum/enchantum/include
    ${DEPS}/spdlog/include ${DEPS}/tt-logger/include ${DEPS}/reflect)
add_executable(test_system_health ${SOURCE}/tests/tt_metal/tt_fabric/system_health/test_system_health.cpp)
add_executable(fabric_smoke_tests ${SOURCE}/tests/tt_metal/tt_fabric/fabric_data_movement/test_basic_fabric_smoke.cpp)
foreach(target test_system_health fabric_smoke_tests)
    target_include_directories(${target} PRIVATE ${INCLUDES})
    target_compile_definitions(${target} PRIVATE SPDLOG_FMT_EXTERNAL FMT_HEADER_ONLY NTEST)
    target_compile_options(${target} PRIVATE -Wno-narrowing)
    target_link_libraries(${target} PRIVATE gmock_main ${LIB}/libtt_metal.so ${LIB}/libtt_stl.so ${LIB}/libtt-umd.so pthread)
    target_link_options(${target} PRIVATE "LINKER:-rpath-link,${LIB}" "LINKER:-rpath-link,${RUNTIME}/../ttnn.libs")
    set_target_properties(${target} PROPERTIES BUILD_RPATH "${LIB};${RUNTIME}/../ttnn.libs")
endforeach()
