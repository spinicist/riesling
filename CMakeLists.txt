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
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON) # Why not?
# Deal with GCC 8 and std::filesystem
# Thanks to Deniz Bahadir on the CMake Discourse
add_link_options("$<$<CXX_COMPILER_ID:GNU>:LINKER:--as-needed>")
link_libraries("$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:-lstdc++fs>")

project(riesling LANGUAGES CXX)
include(Version)
add_definitions(-DEIGEN_USE_THREADS) # Ensure this is always defined across all .cpp files
add_subdirectory(cxx)
