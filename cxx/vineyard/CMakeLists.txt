find_package(Threads REQUIRED)
find_package(args CONFIG REQUIRED)
find_package(ducc CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(hdf5 CONFIG REQUIRED)
find_package(NIFTI CONFIG REQUIRED)
find_package(scn CONFIG REQUIRED)
find_package(tl-ranges CONFIG REQUIRED)

add_library(vineyard
    apodize.cpp
    colors.cpp
    compressor.cpp
    fft.cpp
    filter.cpp
    interp.cpp
    log.cpp
    mapping.cpp
    pad.cpp
    parse_args.cpp
    patches.cpp
    precon.cpp
    regularizers.cpp
    scaling.cpp
    signals.cpp
    subgrid.cpp
    threads.cpp
    trajectory.cpp
    traj_spirals.cpp

    algo/admm.cpp
    algo/bidiag.cpp
    algo/decomp.cpp
    algo/eig.cpp
    algo/gs.cpp
    algo/lad.cpp
    algo/lsmr.cpp
    algo/lsqr.cpp
    algo/otsu.cpp
    algo/pdhg.cpp
    algo/stats.cpp

    basis/basis.cpp
    basis/bernstein.cpp
    basis/fourier.cpp
    basis/svd.cpp

    func/dict.cpp
    func/diffs.cpp

    io/hd5-core.cpp
    io/nifti.cpp
    io/reader.cpp
    io/writer.cpp

    kernel/kernel-f1.cpp
    kernel/kernel-f2.cpp
    kernel/kernel-f3.cpp
    kernel/kernel-cx1.cpp
    kernel/kernel-cx2.cpp
    kernel/kernel-cx3.cpp

    op/fft.cpp
    op/grad.cpp
    op/grid.cpp
    op/hankel.cpp
    op/ndft.cpp
    op/nufft.cpp
    op/op.cpp
    op/ops.cpp
    op/pad.cpp
    op/recon.cpp
    op/reshape.cpp
    op/sense.cpp
    op/top.cpp
    op/wavelets.cpp

    phantom/gradcubes.cpp
    phantom/sphere.cpp
    phantom/shepp-logan.cpp

    prox/entropy.cpp
    prox/hermitian.cpp
    prox/l1-wavelets.cpp
    prox/lsq.cpp
    prox/llr.cpp
    prox/norms.cpp
    prox/prox.cpp
    prox/slr.cpp
    prox/stack.cpp

    sense/coils.cpp
    # sense/espirit.cpp
    sense/sense.cpp

    sim/parameter.cpp
    sim/sequence.cpp
    sim/prep.cpp
    sim/ir.cpp
    sim/dir.cpp
    sim/t2flair.cpp
    sim/t2prep.cpp
)
target_include_directories(vineyard PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}>)
target_link_libraries(vineyard PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    ducc::ducc
    Eigen3::Eigen
    fmt::fmt
    hdf5::hdf5-static
    hdf5::hdf5_hl-static
    NIFTI::niftiio
    tl::ranges
    scn::scn
)
set_target_properties(vineyard PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
)
set_source_files_properties(
    op/grid.cpp
    kernel/kernel-f1.cpp
    kernel/kernel-f2.cpp
    kernel/kernel-f3.cpp
    kernel/kernel-cx1.cpp
    kernel/kernel-cx2.cpp
    kernel/kernel-cx3.cpp
    PROPERTIES COMPILE_FLAGS -ffast-math
)
install(TARGETS vineyard EXPORT vineyard-targets
    LIBRARY DESTINATION  "lib"
    INCLUDES DESTINATION "include/vineyard")
install(DIRECTORY "${CMAKE_CURRENT_LIST_DIR}"
    DESTINATION "include"
    FILES_MATCHING PATTERN "*.hpp"
)
install(EXPORT vineyard-targets
    FILE        "vineyard-targets.cmake"
    NAMESPACE   "vineyard::"
    DESTINATION "share/vineyard"
)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/vineyard-config.cmake "
include(\${CMAKE_CURRENT_LIST_DIR}/vineyard-targets.cmake)
")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/vineyard-config.cmake DESTINATION share/vineyard)
