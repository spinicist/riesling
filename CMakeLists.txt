cmake_minimum_required(VERSION 3.10)
project(riesling LANGUAGES CXX)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR}/cmake)
include(BuildType)
include(Version)

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
    src/espirit.cpp
    src/filter.cpp
    src/kernel.cpp
    src/log.cpp
    src/mapping.cpp
    src/phantom_sphere.cpp
    src/phantom_shepplogan.cpp
    src/parse_args.cpp
    src/sdc.cpp
    src/sense.cpp
    src/threads.cpp
    src/trajectory.cpp
    src/traj_spirals.cpp
    src/zin-grappa.cpp
    src/zin-slr.cpp
    src/algo/decomp.cpp
    src/algo/llr.cpp
    src/fft/fft.cpp
    src/io/hd5-core.cpp
    src/io/nifti.cpp
    src/io/reader.cpp
    src/io/writer.cpp
    src/op/gridBase.cpp
    src/op/grid11.cpp
    src/op/grid31.cpp
    src/op/grid33.cpp
    src/op/grid51.cpp
    src/op/grid55.cpp
    src/op/scaling.cpp
    src/op/sense.hpp
    src/precond/single.cpp
    src/sim/parameter.cpp
    src/sim/dwi.cpp
    src/sim/mprage.cpp
    src/sim/dir.cpp
    src/sim/t2flair.cpp
    src/sim/t2prep.cpp
    src/sim/t1t2.cpp
)
target_include_directories(vineyard PUBLIC src)
target_link_libraries(vineyard PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    Eigen3::Eigen
    fmt::fmt-header-only
    FFTW3::fftw3f
    hdf5::hdf5-static
    NIFTI::niftiio
    scn::scn
)

# Main executable
add_executable(riesling
    src/cmd/admm.cpp
    src/cmd/blend.cpp
    src/cmd/cg.cpp
    src/cmd/compress.cpp
    src/cmd/downsamp.cpp
    src/cmd/espirit.cpp
    src/cmd/grid.cpp
    src/cmd/h5.cpp
    src/cmd/hdr.cpp
    src/cmd/lookup.cpp
    src/cmd/lsmr.cpp
    src/cmd/lsqr.cpp
    src/cmd/meta.cpp
    src/cmd/nii.cpp
    src/cmd/nufft.cpp
    src/cmd/pad.cpp
    src/cmd/phantom.cpp
    src/cmd/plan.cpp
    src/cmd/recon.cpp
    src/cmd/reg.cpp
    src/cmd/sdc.cpp
    src/cmd/sense.cpp
    src/cmd/sense_calib.cpp
    src/cmd/sim.cpp
    src/cmd/split.cpp
    src/cmd/tgv.cpp
    src/cmd/traj.cpp
    src/cmd/version.cpp
    src/cmd/zinfandel.cpp
    src/main.cpp
)
target_link_libraries(riesling
    vineyard
)
set_target_properties(vineyard riesling PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)
install(TARGETS riesling RUNTIME DESTINATION bin)

option(BUILD_TESTS "Build the tests" ON)
if( ${BUILD_TESTS} )
    add_executable(riesling-tests
        test/main.cpp
        test/cropper.cpp
        test/decomp.cpp
        test/fft3.cpp
        test/io.cpp
        test/kernel.cpp
        test/parameters.cpp
        test/sdc.cpp
        test/zinfandel.cpp
        test/op/fft.cpp
        test/op/grid.cpp
        test/op/nufft.cpp
        test/op/pad.cpp
        test/op/sense.cpp
    )
    target_link_libraries(riesling-tests PUBLIC
        vineyard
        Catch2::Catch2
    )
    set_target_properties(vineyard riesling-tests PROPERTIES
        CXX_STANDARD 20
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
        bench/dot.cpp
        bench/kernel.cpp
    )
    target_link_libraries(riesling-bench PUBLIC
        vineyard
        Catch2::Catch2
    )
    set_target_properties(vineyard riesling-bench PROPERTIES
        CXX_STANDARD 20
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
