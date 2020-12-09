cmake_minimum_required(VERSION 3.16)

project(riesling)

# Uncomment for optional debugging
#set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/CMake)
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} ${PROJECT_SOURCE_DIR}/CMake)
include(CheckCXXCompilerFlag)
include(CMakeToolsHelpers OPTIONAL)
include(CMakePrintHelpers)
include( ${PROJECT_SOURCE_DIR}/CMake/BuildType.cmake )

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED on)
include(vcpkg/scripts/cmake/my_flags_sanitizers.cmake)

string(APPEND CMAKE_C_FLAGS ${MY_FLAGS})
string(APPEND CMAKE_CXX_FLAGS ${MY_FLAGS})
string(APPEND CMAKE_C_FLAGS_DEBUG ${MY_FLAGS_DEBUG})
string(APPEND CMAKE_CXX_FLAGS_DEBUG ${MY_FLAGS_DEBUG})

find_package(Threads REQUIRED)
find_package(args CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)
find_package(hdf5 CONFIG REQUIRED)
find_package(ITK 5.0.0 REQUIRED
              COMPONENTS
                ITKCommon
                ITKIOImageBase
                ITKIONIFTI
            )
include( ${ITK_USE_FILE} )

# Ensure this is always defined across all .cpp files
add_compile_definitions(EIGEN_USE_THREADS)

# Common library between tests and main executable
add_library(vineyard
    src/cg.cpp
    src/compressor.cpp
    src/cropper.cpp
    src/fft.cpp
    src/filter.cpp
    src/gridder.cpp
    src/hd5.cpp
    src/interpolator.cpp
    src/io_hd5.cpp
    src/io_nifti.cpp
    src/kaiser-bessel.cpp
    src/log.cpp
    src/parse_args.cpp
    src/sense.cpp
    src/slab_correct.cpp
    src/tgv.cpp
    src/threads.cpp
    src/traj_archimedean.cpp
    src/wavelets.cpp
    src/zinfandel.cpp
)


# hdf5::hdf5-static hdf5::hdf5_hl-static hdf5::hdf5_cpp-static hdf5::hdf5_hl_cpp-static
target_link_libraries(vineyard PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    Eigen3::Eigen
    fmt::fmt
    FFTW3::fftw3f
    hdf5::hdf5-static
    ${ITK_LIBRARIES}
)

# Main executable
add_executable(riesling
    src/main_hdr.cpp    
    src/main_phantom.cpp
    src/main_compress.cpp
    src/main_zinfandel.cpp
    src/main_kspace.cpp
    src/main_traj.cpp
    src/main_rss.cpp
    src/main_sense.cpp
    src/main_tgv.cpp
    src/main_toeplitz.cpp
    src/main_ds.cpp
    src/main.cpp
)
target_link_libraries(riesling
    vineyard
)
install(TARGETS riesling RUNTIME DESTINATION bin)

# Tests
add_executable(prufung
    test/main.cpp
    test/fft.cpp
    test/zinfandel.cpp
)
target_link_libraries(prufung PUBLIC
    vineyard
    Catch2::Catch2
)

# Documentation, which is admittedly sparse
option(BUILD_DOC "Build documentation" OFF)
find_package(Doxygen)
if (DOXYGEN_FOUND)
    set( DOXYGEN_OUTPUT_DIRECTORY doc )
    set( DOXYGEN_COLLABORATION_GRAPH YES )
    set( DOXYGEN_EXTRACT_ALL YES )
    set( DOXYGEN_CLASS_DIAGRAMS YES )
    set( DOXYGEN_HIDE_UNDOC_RELATIONS NO )
    set( DOXYGEN_HAVE_DOT YES )
    set( DOXYGEN_CLASS_GRAPH YES )
    set( DOXYGEN_CALL_GRAPH YES )
    set( DOXYGEN_CALLER_GRAPH YES )
    set( DOXYGEN_COLLABORATION_GRAPH YES )
    set( DOXYGEN_BUILTIN_STL_SUPPORT YES )
    set( DOXYGEN_EXTRACT_PRIVATE YES )
    set( DOXYGEN_EXTRACT_PACKAGE YES )
    set( DOXYGEN_EXTRACT_STATIC YES )
    set( DOXYGEN_EXTRACT_LOCALMETHODS YES )
    set( DOXYGEN_UML_LOOK YES )
    set( DOXYGEN_UML_LIMIT_NUM_FIELDS 50 )
    set( DOXYGEN_TEMPLATE_RELATIONS YES )
    set( DOXYGEN_DOT_GRAPH_MAX_NODES 100 )
    set( DOXYGEN_MAX_DOT_GRAPH_DEPTH 0 )
    set( DOXYGEN_DOT_TRANSPARENT YES )
    doxygen_add_docs(doc ${CMAKE_CURRENT_SOURCE_DIR}/src)
else (DOXYGEN_FOUND)
    message("Doxygen required to build the documentation")
endif (DOXYGEN_FOUND)