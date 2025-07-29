option(BUILD_TESTS "Build the tests" ON)
if(${BUILD_TESTS})
    find_package(Catch2 CONFIG REQUIRED)
    add_executable(riesling-tests
        algo.cpp
        decomp.cpp
        fft1.cpp
        fft3.cpp
        io.cpp
        kernel.cpp
        patches.cpp
        precon.cpp
        prox.cpp
        op/fft.cpp
        # op/grad.cpp
        op/grid.cpp
        # op/ndft.cpp
        op/nufft.cpp
        op/pad.cpp
        op/recon.cpp
        op/sense.cpp
        op/wavelets.cpp
    )
    target_link_libraries(riesling-tests PUBLIC
        rl
        Catch2::Catch2WithMain
    )
    target_compile_definitions(riesling-tests PRIVATE
        -DCATCH_CONFIG_CONSOLE_WIDTH=200
    )
    set_target_properties(riesling-tests PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
    )
endif()
