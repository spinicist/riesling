find_package(Threads REQUIRED)
find_package(ducc CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(flux CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(hdf5 CONFIG REQUIRED)
find_package(NIFTI CONFIG REQUIRED)
find_package(scn CONFIG REQUIRED)

set(SRC_FILES
colors.cpp
compressor.cpp
fft.cpp
filter.cpp
interp.cpp
patches.cpp
precon.cpp
scaling.cpp
slice.cpp
trajectory.cpp

algo/admm.cpp
algo/bidiag.cpp
# algo/cg.cpp
algo/decomp.cpp
algo/eig.cpp
algo/gs.cpp
algo/iter.cpp
algo/lsmr.cpp
algo/lsqr.cpp
algo/otsu.cpp
algo/pdhg.cpp
algo/stats.cpp

basis/basis.cpp
basis/bernstein.cpp
basis/svd.cpp

func/dict.cpp
func/diffs.cpp

io/hd5-core.cpp
io/nifti.cpp
io/reader.cpp
io/writer.cpp

log/log.cpp
log/debug.cpp

op/apodize.cpp
op/f0.cpp
op/fft.cpp
op/grad.cpp
op/grid.cpp
op/grid-decant.cpp
# op/hankel.cpp
op/laplacian.cpp
op/mask.cpp
# op/ndft.cpp
op/nufft.cpp
op/nufft-decant.cpp
op/nufft-lowmem.cpp
op/op.cpp
op/ops.cpp
op/pad.cpp
op/recon.cpp
op/reshape.cpp
op/sense.cpp
op/top-id.cpp
op/wavelets.cpp

phantom/cartesian.cpp
phantom/gradcubes.cpp
phantom/sphere.cpp
phantom/shepp-logan.cpp
phantom/radial.cpp

prox/l1-wavelets.cpp
prox/llr.cpp
prox/norms.cpp
prox/prox.cpp
# prox/slr.cpp
prox/stack.cpp

sense/coils.cpp
sense/sense.cpp

sim/ir.cpp
sim/dir.cpp
sim/t2flair.cpp
sim/t2prep.cpp
sim/zte.cpp

sys/signals.cpp
sys/threads.cpp
)

set(HDR_FILES
types.hpp

colors.hpp
compressor.hpp
fft.hpp
filter.hpp
interp.hpp
info.hpp
patches.hpp
precon.hpp
scaling.hpp
slice.hpp
tensors.hpp
trajectory.hpp

algo/admm.hpp
algo/bidiag.hpp
algo/cg.hpp
algo/decomp.hpp
algo/eig.hpp
algo/gs.hpp
algo/iter.hpp
algo/lsmr.hpp
algo/lsqr.hpp
algo/otsu.hpp
algo/pdhg.hpp
algo/stats.hpp

basis/basis.hpp
basis/bernstein.hpp
basis/svd.hpp

func/dict.hpp
func/diffs.hpp

io/hd5-core.hpp
io/nifti.hpp
io/reader.hpp
io/writer.hpp

log/log.hpp
log/debug.hpp

op/apodize.hpp
op/f0.hpp
op/fft.hpp
op/grad.hpp
op/grid.hpp
op/grid-decant.hpp
op/hankel.hpp
op/ndft.hpp
op/nufft.hpp
op/nufft-decant.hpp
op/nufft-lowmem.hpp
op/op.hpp
op/ops.hpp
op/pad.hpp
op/recon.hpp
op/reshape.hpp
op/sense.hpp
op/wavelets.hpp

phantom/gradcubes.hpp
phantom/sphere.hpp
phantom/shepp-logan.hpp

prox/hermitian.hpp
prox/l1-wavelets.hpp
# prox/lsq.hpp
prox/llr.hpp
prox/norms.hpp
prox/prox.hpp
prox/slr.hpp
prox/stack.hpp

sense/coils.hpp
# sense/espirit.cpp
sense/sense.hpp

sim/ir.hpp
sim/dir.hpp
sim/t2flair.hpp
sim/t2prep.hpp
sim/zte.hpp

sys/signals.hpp
sys/threads.hpp
)
add_library(rl
    ${SRC_FILES}
)
target_sources(rl PUBLIC FILE_SET HEADERS
    TYPE HEADERS
    FILES ${HDR_FILES}
    BASE_DIRS ${CMAKE_SOURCE_DIR}/cxx
)

target_include_directories(rl PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/cxx>
  $<INSTALL_INTERFACE:include/rl>
)
target_link_libraries(rl PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
    ducc::ducc
    Eigen3::Eigen
    flux::flux
    fmt::fmt
    hdf5::hdf5-static
    hdf5::hdf5_hl-static
    NIFTI::niftiio
    scn::scn
)
set_target_properties(rl PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
)
set_source_files_properties(
    op/ops.cpp
    op/ndft.cpp
    op/grid.cpp
    op/grid-subgrid.cpp
    kernel/kernel-f1.cpp
    kernel/kernel-f2.cpp
    kernel/kernel-f3.cpp
    kernel/kernel-cx1.cpp
    kernel/kernel-cx2.cpp
    kernel/kernel-cx3.cpp
    PROPERTIES COMPILE_FLAGS -ffast-math
)
install(TARGETS rl EXPORT rl-targets
    FILE_SET HEADERS
)
install(EXPORT rl-targets
    FILE        "rl-targets.cmake"
    NAMESPACE   "rl::"
    DESTINATION "share/rl"
)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/rl-config.cmake "
include(\${CMAKE_CURRENT_LIST_DIR}/rl-targets.cmake)

find_package(Threads REQUIRED)
find_package(ducc CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(flux CONFIG REQUIRED)
find_package(fmt CONFIG REQUIRED)
find_package(hdf5 CONFIG REQUIRED)
find_package(NIFTI CONFIG REQUIRED)
find_package(scn CONFIG REQUIRED)
")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/rl-config.cmake DESTINATION share/rl)
