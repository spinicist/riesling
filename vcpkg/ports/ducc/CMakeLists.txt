cmake_minimum_required(VERSION 3.10)
# set the project name
project(DUCC)

# Set the package name
set(PKGNAME "ducc")
# Set the version
find_package(Git)
if (GIT_FOUND)
    execute_process(
            COMMAND ${GIT_EXECUTABLE} describe --abbrev=0 --tags
            WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
            OUTPUT_VARIABLE VERSION
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else ()
    message(WARNING "Git not found! Using default version unknown")
    set(VERSION "unknown")
endif ()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if (NOT MSVC)
    if (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc|ppc64|powerpc|powerpc64" OR (APPLE AND CMAKE_OSX_ARCHITECTURES MATCHES "ppc|ppc64"))
        # PowerPC arch does not have -march flag.
        set(DUCC0_ARCH_FLAGS "-mtune=native" CACHE STRING "Compiler flags for specifying target architecture.")
    else ()
        set(DUCC0_ARCH_FLAGS "-march=native" CACHE STRING "Compiler flags for specifying target architecture.")
    endif ()
    message(STATUS "Using GCC/Clang flags: ${DUCC0_ARCH_FLAGS}")
else ()
    # Check for AVX, AVX512 and SSE support
    message(STATUS "Checking for AVX, AVX512 and SSE support")
    try_run(RUN_RESULT_VAR COMPILE_RESULT_VAR
            ${CMAKE_BINARY_DIR}
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckAVX.cpp
            COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT
            RUN_OUTPUT_VARIABLE RUN_OUTPUT)
    if (RUN_OUTPUT MATCHES "AVX512")
        set(DUCC0_ARCH_FLAGS "/arch:AVX512" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (RUN_OUTPUT MATCHES "AVX")
        set(DUCC0_ARCH_FLAGS "/arch:AVX" CACHE STRING "Compiler flags for specifying target architecture.")
    elseif (RUN_OUTPUT MATCHES "SSE")
        set(DUCC0_ARCH_FLAGS "/arch:SSE" CACHE STRING "Compiler flags for specifying target architecture.")
    else ()
        set(DUCC0_ARCH_FLAGS "" CACHE STRING "Compiler flags for specifying target architecture.")
    endif ()
    message(STATUS "CPU supports: ${RUN_OUTPUT}")
    message(STATUS "Using MSVC flags: ${DUCC0_ARCH_FLAGS}")
endif ()

# It is possible to add a sphinx documentation here
option(DUCC0_USE_THREADS "Use threads for parallelization" ON)

add_library(ducc STATIC
        src/ducc0/healpix/healpix_base.cc
        src/ducc0/healpix/healpix_tables.cc
        src/ducc0/math/gl_integrator.cc
        src/ducc0/math/pointing.cc
        src/ducc0/math/gridding_kernel.cc
        src/ducc0/math/geom_utils.cc
        src/ducc0/math/wigner3j.cc
        src/ducc0/math/space_filling.cc
        src/ducc0/wgridder/wgridder.cc
        src/ducc0/infra/string_utils.cc
        src/ducc0/infra/communication.cc
        src/ducc0/infra/types.cc
        src/ducc0/infra/system.cc
        src/ducc0/infra/threading.cc
        src/ducc0/infra/mav.cc
        # add more source files here
)

target_compile_features(ducc PRIVATE cxx_std_17)

set(HDR_FILES
src/ducc0/bindings/array_descriptor.h
src/ducc0/bindings/pybind_utils.h
src/ducc0/bindings/typecode.h
src/ducc0/fft/fft_inst_inc.h
src/ducc0/fft/fft.h
src/ducc0/fft/fft1d_impl.h
src/ducc0/fft/fftnd_impl.h
src/ducc0/healpix/healpix_base.h
src/ducc0/healpix/healpix_tables.h
src/ducc0/infra/aligned_array.h
src/ducc0/infra/bucket_sort.h
src/ducc0/infra/communication.h
src/ducc0/infra/error_handling.h
src/ducc0/infra/mav.h
src/ducc0/infra/misc_utils.h
src/ducc0/infra/simd.h
src/ducc0/infra/string_utils.h
src/ducc0/infra/system.h
src/ducc0/infra/threading.h
src/ducc0/infra/timers.h
src/ducc0/infra/types.h
src/ducc0/infra/useful_macros.h
src/ducc0/math/cmplx.h
src/ducc0/math/constants.h
src/ducc0/math/geom_utils.h
src/ducc0/math/gl_integrator.h
src/ducc0/math/gridding_kernel.h
src/ducc0/math/math_utils.h
src/ducc0/math/mcm.h
src/ducc0/math/pointing.h
src/ducc0/math/quaternion.h
src/ducc0/math/rangeset.h
src/ducc0/math/solvers.h
src/ducc0/math/space_filling.h
src/ducc0/math/unity_roots.h
src/ducc0/math/vec3.h
src/ducc0/math/wigner3j.h
src/ducc0/nufft/nufft_common.h
src/ducc0/nufft/nufft.h
src/ducc0/nufft/spreadinterp_impl.h
src/ducc0/nufft/spreadinterp.h
src/ducc0/sht/alm.h
src/ducc0/sht/sht_utils.h
src/ducc0/sht/sht.h
src/ducc0/sht/sphere_interpol.h
src/ducc0/sht/totalconvolve.h
src/ducc0/wgridder/wgridder_impl.h
src/ducc0/wgridder/wgridder_inst_inc.h
src/ducc0/wgridder/wgridder.h)
target_sources(ducc PUBLIC FILE_SET HEADERS
    TYPE HEADERS
    BASE_DIRS "src"
    FILES ${HDR_FILES}
)

target_compile_options(ducc PRIVATE
        -ffp-contract=fast
        -fexcess-precision=fast
        -fno-math-errno
        -fno-signed-zeros
        -fno-trapping-math
        -fassociative-math
        -freciprocal-math
        # if nan are not used we could disable them manually
        # Why no -fast-math or -funsafe-math-optimizations ?
        # It breaks the code, and influences other software that depend on ducc by changing the rounding mode.
        # GCC-13 recently fixed this issue:
        # https://github.com/llvm/llvm-project/issues/57589
        # https://gcc.gnu.org/gcc-13/changes.html
        # https://trofi.github.io/posts/302-Ofast-and-ffast-math-non-local-effects.html

)
target_compile_options(ducc PRIVATE SHELL:$<$<CONFIG:Release,RelWithDebInfo>:${DUCC0_ARCH_FLAGS}>)


target_compile_definitions(ducc PRIVATE PKGNAME=${PKGNAME} PKGVERSION=${VERSION})
if(MSVC)
    target_compile_options(ducc PRIVATE
            /W4 # Equivalent of -Wall
            /WX # Equivalent of -Wfatal-errors
            # Add other MSVC specific flags here
    )
endif()
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ducc PRIVATE
            -Wfatal-errors
            -Wfloat-conversion
            -W
            -Wall
            -Wstrict-aliasing
            -Wwrite-strings
            -Wredundant-decls
            -Woverloaded-virtual
            -Wcast-qual
            -Wcast-align
            -Wpointer-arith
            -Wnon-virtual-dtor
            -Wzero-as-null-pointer-constant
    )
endif()

find_package(Threads REQUIRED)
target_link_libraries(ducc PRIVATE Threads::Threads)

install(TARGETS ducc
        EXPORT ${PKGNAME}-targets
        FILE_SET HEADERS
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
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

