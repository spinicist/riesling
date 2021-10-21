cmake_minimum_required(VERSION 3.10)
project(riesling LANGUAGES CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(BuildType)
include(GitVersion)

# Deal with GCC 8 and std::filesystem
add_link_options( "$<$<CXX_COMPILER_ID:GNU>:LINKER:--as-needed>" )
# Thanks to Deniz Bahadir on the CMake Discourse
link_libraries( "$<$<AND:$<CXX_COMPILER_ID:GNU>,$<VERSION_LESS:$<CXX_COMPILER_VERSION>,9.0>>:-lstdc++fs>" )

find_package(Threads REQUIRED)
find_package(args REQUIRED)
find_package(Catch2 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(fmt REQUIRED)
find_package(FFTW3f REQUIRED)
find_package(hdf5 CONFIG REQUIRED)
find_package(NIFTI REQUIRED)
find_package(scn REQUIRED)

# Ensure this is always defined across all .cpp files
add_definitions(-DEIGEN_USE_THREADS)

# Common library between tests and main executable
add_library(vineyard
    src/coils.cpp
    src/compressor.cpp
    src/cropper.cpp
    src/decomp.cpp
    src/espirit.cpp
    src/fft1.cpp
    src/fft_plan.cpp
    src/fft_util.cpp
    src/filter.cpp
    src/hankel.cpp
    src/io_hd5.cpp
    src/io_nifti.cpp
    src/llr.cpp
    src/log.cpp
    src/padder.cpp
    src/phantom_sphere.cpp
    src/phantom_shepplogan.cpp
    src/parse_args.cpp
    src/sdc.cpp
    src/sense.cpp
    src/sim.cpp
    src/sim-prep.cpp
    src/sim-eddy.cpp
    src/slab_correct.cpp
    src/tgv.cpp
    src/threads.cpp
    src/trajectory.cpp
    src/traj_archimedean.cpp
    src/vc.cpp
    src/wavelets.cpp
    src/zinfandel.cpp
    src/op/operator.cpp
    src/op/crop.cpp
    src/op/grid.cpp
    src/op/grid-nn.cpp
    src/op/grid-kb.cpp
    src/op/grid-basis.cpp
    src/op/grid-basis-nn.cpp
    src/op/grid-basis-kb.cpp
    src/op/recon.cpp
    src/op/recon-basis.cpp
    src/op/sense.cpp
)
target_link_libraries(vineyard PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    Eigen3::Eigen
    fmt::fmt
    FFTW3::fftw3f
    hdf5::hdf5-static
    NIFTI::niftiio
    scn::scn
)

# Main executable
add_executable(riesling
    src/main_basis_admm.cpp
    src/main_basis_blend.cpp
    src/main_basis_dict.cpp
    src/main_basis_recon.cpp
    src/main_basis_sim.cpp
    src/main_cg.cpp
    src/main_cgvar.cpp
    src/main_compress.cpp
    src/main_espirit.cpp
    src/main_grid.cpp
    src/main_hdr.cpp
    src/main_phantom.cpp
    src/main_plan.cpp
    src/main_recon.cpp
    src/main_tgv.cpp
    src/main_traj.cpp
    src/main_sdc.cpp
    src/main_sense.cpp
    src/main_split.cpp
    src/main_version.cpp
    src/main_zinfandel.cpp
    src/main.cpp
)
target_link_libraries(riesling
    vineyard
)
set_target_properties(vineyard riesling PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
install(TARGETS riesling RUNTIME DESTINATION bin)
# Parse version number
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/version.h.in ${CMAKE_CURRENT_BINARY_DIR}/version.h)
target_include_directories(riesling PRIVATE ${CMAKE_CURRENT_BINARY_DIR})

option(BUILD_TESTS "Build the tests" ON)
if( ${BUILD_TESTS} )
    add_executable(riesling-tests
        test/main.cpp
        test/cropper.cpp
        test/decomp.cpp
        test/fft1.cpp
        test/fft3.cpp
        test/gridder.cpp
        test/hankel.cpp
        test/sdc.cpp
        test/zinfandel.cpp
        test/op/crop.cpp
        test/op/grid.cpp
        test/op/sense.cpp
    )
    target_link_libraries(riesling-tests PUBLIC
        vineyard
        Catch2::Catch2
    )
    set_target_properties(vineyard riesling-tests PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endif()

option(BUILD_BENCHMARKS "Build the benchmarks" OFF)
if ( ${BUILD_BENCHMARKS} )
    add_executable(riesling-bench
        bench/main.cpp
        bench/rss.cpp
        bench/grid.cpp
    )
    target_link_libraries(riesling-bench PUBLIC
        vineyard
        Catch2::Catch2
    )
    set_target_properties(vineyard riesling-bench PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
    )
endif()

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
