cmake_minimum_required(VERSION 3.13)
project(riesling LANGUAGES CXX)
include(Version)

# Ensure this is always defined across all .cpp files
add_definitions(-DEIGEN_USE_THREADS)

add_subdirectory(cxx)
