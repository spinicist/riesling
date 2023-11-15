cmake_minimum_required(VERSION 3.13)
project(riesling LANGUAGES CXX)
include(Version)

find_package(Threads REQUIRED)
find_package(args CONFIG REQUIRED)
find_package(Catch2 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(FFTW3f CONFIG REQUIRED)
find_package(hdf5 CONFIG REQUIRED)
find_package(NIFTI CONFIG REQUIRED)
find_package(range-v3 CONFIG REQUIRED)
find_package(scn CONFIG REQUIRED)

# Ensure this is always defined across all .cpp files
add_definitions(-DEIGEN_USE_THREADS)

# Common library between tests and main executable
add_library(vineyard
    src/basis.cpp
    src/compressor.cpp
    src/cropper.cpp
    src/filter.cpp
    src/interp.cpp
    src/log.cpp
    src/mapping.cpp
    src/parse_args.cpp
    src/patches.cpp
    src/precond.cpp
    src/regularizers.cpp
    src/scaling.cpp
    src/sdc.cpp
    src/signals.cpp
    src/threads.cpp
    src/trajectory.cpp
    src/traj_spirals.cpp
    src/zin-grappa.cpp
    src/algo/admm.cpp
    src/algo/bidiag.cpp
    src/algo/cg.cpp
    src/algo/decomp.cpp
    src/algo/eig.cpp
    src/algo/lsmr.cpp
    src/algo/lsqr.cpp
    src/algo/otsu.cpp
    src/algo/pdhg.cpp
    src/algo/stats.cpp
    src/fft/fft.cpp
    src/func/dict.cpp
    src/func/diffs.cpp
    src/img/rgb.cpp
    src/io/hd5-core.cpp
    src/io/nifti.cpp
    src/io/reader.cpp
    src/io/writer.cpp
    src/kernel/kernel.cpp
    src/op/apodize.cpp
    src/op/grad.cpp
    src/op/grid.cpp
    src/op/ndft.cpp
    src/op/nufft.cpp
    src/op/ops.cpp
    src/op/pad.cpp
    src/op/recon.cpp
    src/op/sense.cpp
    src/op/wavelets.cpp
    src/phantom/gradcubes.cpp
    src/phantom/sphere.cpp
    src/phantom/shepp-logan.cpp
    src/prox/entropy.cpp
    src/prox/l1-wavelets.cpp
    src/prox/lsq.cpp
    src/prox/llr.cpp
    src/prox/norms.cpp
    src/prox/prox.cpp
    src/prox/slr.cpp
    src/prox/stack.cpp
    src/sense/coils.cpp
    src/sense/espirit.cpp
    src/sense/rovir.cpp
    src/sense/sense.cpp
    src/sim/parameter.cpp
    src/sim/dir.cpp
    src/sim/ir.cpp
    src/sim/t2flair.cpp
    src/sim/t2prep.cpp
)
target_include_directories(vineyard PUBLIC src)
target_link_libraries(vineyard PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    Eigen3::Eigen
    fmt::fmt-header-only
    FFTW3::fftw3f
    hdf5::hdf5-static
    NIFTI::niftiio
    range-v3
    scn::scn
)
set_target_properties(vineyard PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
)

# Main executable
add_executable(riesling
    src/cmd/admm.cpp
    src/cmd/basis-img.cpp
    src/cmd/basis-sim.cpp
    src/cmd/blend.cpp
    src/cmd/cg.cpp
    src/cmd/compress.cpp
    src/cmd/denoise.cpp
    src/cmd/downsamp.cpp
    src/cmd/eig.cpp
    src/cmd/filter.cpp
    src/cmd/frames.cpp
    src/cmd/grad.cpp
    src/cmd/grid.cpp
    src/cmd/h5.cpp
    # src/cmd/lad.cpp
    src/cmd/lookup.cpp
    src/cmd/lsmr.cpp
    src/cmd/lsqr.cpp
    src/cmd/merge.cpp
    src/cmd/nii.cpp
    src/cmd/noisify.cpp
    src/cmd/ndft.cpp
    src/cmd/nufft.cpp
    src/cmd/pad.cpp
    src/cmd/pdhg.cpp
    src/cmd/pdhg-setup.cpp
    src/cmd/phantom.cpp
    src/cmd/plan.cpp
    src/cmd/precond.cpp
    src/cmd/prox.cpp
    src/cmd/recon-rss.cpp
    src/cmd/recon-sense.cpp
    src/cmd/sdc.cpp
    src/cmd/sense.cpp
    src/cmd/sense-calib.cpp
    src/cmd/sense-sim.cpp
    src/cmd/slice.cpp
    src/cmd/traj.cpp
    src/cmd/version.cpp
    src/cmd/wavelets.cpp
    src/cmd/zinfandel.cpp
    src/main.cpp
)
target_link_libraries(riesling PRIVATE
    vineyard
)
set_target_properties(riesling PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
)
install(TARGETS riesling RUNTIME DESTINATION bin)

option(BUILD_VIEW "Build the view command. Requires notcurses" OFF)

if (${BUILD_VIEW})
    target_sources(riesling PRIVATE src/cmd/view.cpp)
    target_compile_definitions(riesling PRIVATE -DBUILD_VIEW)
    # notcurses CMake is broken, as is compiling via vcpkg
    if (LINUX)
        set(Notcurses_INCLUDE_DIRS ${CMAKE_PREFIX_PATH}/include)
	set(Notcurses_LIBRARY_DIRS ${CMAKE_PREFIX_PATH}/lib64 ${CMAKE_PREFIX_PATH}/lib)
    else()
        set(Notcurses_LIBRARY_DIRS "/opt/homebrew/lib")
        set(Notcurses_INCLUDE_DIRS "/opt/homebrew/include")
    endif()
    set(Notcurses_LIBRARIES
        notcurses-core
        notcurses
    )
    target_include_directories(riesling PRIVATE
        ${Notcurses_INCLUDE_DIRS}
    )
    target_link_directories(riesling PRIVATE
        ${Notcurses_LIBRARY_DIRS}
    )
    target_link_libraries(riesling PRIVATE ${Notcurses_LIBRARIES})
endif()

option(BUILD_TESTS "Build the tests" ON)

if(${BUILD_TESTS})
    add_executable(riesling-tests
        test/algo.cpp
        test/cropper.cpp
        test/decomp.cpp
        test/fft3.cpp
        test/io.cpp
        test/kernel.cpp
        test/parameters.cpp
        test/precond.cpp
        test/sdc.cpp
        test/zinfandel.cpp
        test/op/fft.cpp
        test/op/grid.cpp
        test/op/ndft.cpp
        test/op/nufft.cpp
        test/op/pad.cpp
        test/op/recon.cpp
        test/op/sense.cpp
        test/op/wavelets.cpp
    )
    target_link_libraries(riesling-tests PUBLIC
        vineyard
        Catch2::Catch2WithMain
    )
    set_target_properties(riesling-tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
    )
endif()

option(BUILD_BENCHMARKS "Build the benchmarks" OFF)

if(${BUILD_BENCHMARKS})
    add_executable(riesling-bench
        bench/dot.cpp
        bench/grid.cpp
        bench/kernel.cpp
        bench/rss.cpp
    )
    target_link_libraries(riesling-bench PUBLIC
        vineyard
        Catch2::Catch2WithMain
    )
    target_compile_definitions(riesling-bench PRIVATE
    CATCH_CONFIG_CONSOLE_WIDTH=200
    )
    set_target_properties(riesling-bench PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
    )
endif()

# Documentation, which is admittedly sparse
option(BUILD_DOC "Build documentation" OFF)
find_package(Doxygen)

if(DOXYGEN_FOUND)
    set(DOXYGEN_OUTPUT_DIRECTORY doc)
    set(DOXYGEN_COLLABORATION_GRAPH YES)
    set(DOXYGEN_EXTRACT_ALL YES)
    set(DOXYGEN_CLASS_DIAGRAMS YES)
    set(DOXYGEN_HIDE_UNDOC_RELATIONS NO)
    set(DOXYGEN_HAVE_DOT YES)

    set(DOXYGEN_CLASS_GRAPH YES)
    set(DOXYGEN_CALL_GRAPH YES)
    set(DOXYGEN_CALLER_GRAPH YES)
    set(DOXYGEN_COLLABORATION_GRAPH YES)
    set(DOXYGEN_BUILTIN_STL_SUPPORT YES)
    set(DOXYGEN_EXTRACT_PRIVATE YES)
    set(DOXYGEN_EXTRACT_PACKAGE YES)
    set(DOXYGEN_EXTRACT_STATIC YES)
    set(DOXYGEN_EXTRACT_LOCALMETHODS YES)
    set(DOXYGEN_UML_LOOK YES)
    set(DOXYGEN_UML_LIMIT_NUM_FIELDS 50)
    set(DOXYGEN_TEMPLATE_RELATIONS YES)
    set(DOXYGEN_DOT_GRAPH_MAX_NODES 100)
    set(DOXYGEN_MAX_DOT_GRAPH_DEPTH 0)
    set(DOXYGEN_DOT_TRANSPARENT YES)
    doxygen_add_docs(doc ${CMAKE_CURRENT_SOURCE_DIR}/src)
else(DOXYGEN_FOUND)
    message("Doxygen required to build the documentation")
endif(DOXYGEN_FOUND)
