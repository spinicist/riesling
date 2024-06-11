cmake_minimum_required(VERSION 3.23)
project(ducc LANGUAGES CXX)
find_package(Threads REQUIRED)

set(SRC_FILES
src/ducc0/infra/threading.cc
src/ducc0/infra/string_utils.cc
src/ducc0/infra/communication.cc
src/ducc0/infra/mav.cc
src/ducc0/infra/types.cc
src/ducc0/infra/system.cc
src/ducc0/math/geom_utils.cc
src/ducc0/math/wigner3j.cc
src/ducc0/math/space_filling.cc
src/ducc0/math/pointing.cc
src/ducc0/math/gridding_kernel.cc
src/ducc0/math/gl_integrator.cc
src/ducc0/sht/sht.cc
src/ducc0/wgridder/wgridder.cc
src/ducc0/healpix/healpix_tables.cc
src/ducc0/healpix/healpix_base.cc)
set(HDR_FILES
src/ducc0/infra/timers.h
src/ducc0/infra/string_utils.h
src/ducc0/infra/communication.h
src/ducc0/infra/types.h
src/ducc0/infra/threading.h
src/ducc0/infra/useful_macros.h
src/ducc0/infra/misc_utils.h
src/ducc0/infra/mav.h
src/ducc0/infra/simd.h
src/ducc0/infra/bucket_sort.h
src/ducc0/infra/system.h
src/ducc0/infra/aligned_array.h
src/ducc0/infra/sycl_utils.h
src/ducc0/infra/error_handling.h
src/ducc0/math/geom_utils.h
src/ducc0/math/rangeset.h
src/ducc0/math/space_filling.h
src/ducc0/math/wigner3j.h
src/ducc0/math/unity_roots.h
src/ducc0/math/quaternion.h
src/ducc0/math/solvers.h
src/ducc0/math/gridding_kernel.h
src/ducc0/math/constants.h
src/ducc0/math/cmplx.h
src/ducc0/math/math_utils.h
src/ducc0/math/vec3.h
src/ducc0/math/gl_integrator.h
src/ducc0/math/pointing.h
src/ducc0/nufft/nufft_r.h
src/ducc0/nufft/nufft.h
src/ducc0/sht/sht.h
src/ducc0/sht/alm.h
src/ducc0/sht/sht_utils.h
src/ducc0/sht/sphere_interpol.h
src/ducc0/sht/totalconvolve.h
src/ducc0/wgridder/wgridder.h
src/ducc0/wgridder/wgridder_sycl.h
src/ducc0/fft/fft.h
src/ducc0/fft/fftnd_impl.h
src/ducc0/fft/fft1d_impl.h
src/ducc0/healpix/healpix_tables.h
src/ducc0/healpix/healpix_base.h
src/ducc0/bindings/typecode.h
src/ducc0/bindings/pybind_utils.h
src/ducc0/bindings/array_descriptor.h)
add_library(ducc ${SRC_FILES})
target_sources(ducc PUBLIC FILE_SET HEADERS
    TYPE HEADERS
    BASE_DIRS src
    BASE_DIRS "src"
    FILES ${HDR_FILES}
)
set_target_properties(ducc PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    C_EXTENSIONS ON
    CXX_EXTENSIONS ON
    CXX_VISIBILITY_PRESET hidden
)
target_link_libraries(ducc PUBLIC
    ${CMAKE_THREAD_LIBS_INIT}
)
install(TARGETS ducc EXPORT ducc-targets
        FILE_SET HEADERS
        LIBRARY DESTINATION lib
)
install(
    EXPORT ducc-targets
    FILE ducc-targets.cmake
    NAMESPACE ducc::
    DESTINATION share/ducc
)
file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/ducc-config.cmake "
include(\${CMAKE_CURRENT_LIST_DIR}/ducc-targets.cmake)
")
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ducc-config.cmake DESTINATION share/ducc)
