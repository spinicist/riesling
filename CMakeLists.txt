add_compile_options(-finput-charset=UTF-8)
cmake_minimum_required(VERSION 3.13)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_LIST_DIR}/cmake)
include(CheckCXXCompilerFlag)
include(CMakeToolsHelpers OPTIONAL)
include(CMakePrintHelpers)
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()
if(LINUX)
    set(CMAKE_EXE_LINKER_FLAGS_INIT "-static-libgcc -static-libstdc++")
endif()
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF) # Why not? Because it takes forever, that's why
# Deal with stupid complex ABI warnings from GCC 4.4 onwards
add_compile_options("$<$<CXX_COMPILER_ID:GNU>:-Wno-psabi>")
# Deal with GCC 8 and std::filesystem
# Thanks to Deniz Bahadir on the CMake Discourse
add_link_options("$<$<CXX_COMPILER_ID:GNU>:LINKER:--as-needed>")
link_libraries("$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:-lstdc++fs>")
option(BUILD_MERLIN "Build MERLIN tool" OFF)
if(BUILD_MERLIN)
  list(APPEND VCPKG_MANIFEST_FEATURES "merlin")
endif()
project(riesling LANGUAGES CXX)
include(Version)
add_definitions(-DEIGEN_USE_THREADS -DEIGEN_DONT_PARALLELIZE) # Ensure this is always defined across all .cpp files
add_subdirectory(cxx)
