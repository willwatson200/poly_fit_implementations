cmake_minimum_required(VERSION 3.5.0)
project(poly_fit_cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(inc)


# Test Executable
add_executable(poly_fit ${CMAKE_SOURCE_DIR}/src/poly_fit.cc)
target_link_libraries(poly_fit ${CMAKE_SOURCE_DIR}/lib/cnpy/build/libcnpy.so)
