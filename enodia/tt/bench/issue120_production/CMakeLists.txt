cmake_minimum_required(VERSION 3.22...4.2)
project(issue120_production LANGUAGES CXX)

# Reuse the immutable pinned dependency/HAL admission, not the old executable.
add_subdirectory(../issue120 legacy EXCLUDE_FROM_ALL)
add_executable(issue120_production native.cpp)
add_dependencies(issue120_production issue120_hal_headers)
target_compile_features(issue120_production PRIVATE cxx_std_20)
foreach(property INCLUDE_DIRECTORIES LINK_DIRECTORIES LINK_LIBRARIES LINK_OPTIONS COMPILE_DEFINITIONS BUILD_RPATH)
    get_target_property(inherited issue120_probe ${property})
    if(inherited)
        set_property(TARGET issue120_production PROPERTY ${property} "${inherited}")
    endif()
endforeach()
set_target_properties(issue120_production PROPERTIES BUILD_RPATH_USE_ORIGIN TRUE)
add_custom_command(TARGET issue120_production POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${TT_METAL_LIBRARY}" "$<TARGET_FILE_DIR:issue120_production>/libtt_metal.so")
