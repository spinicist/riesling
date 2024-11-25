option(BUILD_BENCHMARKS "Build the benchmarks" OFF)
if(${BUILD_BENCHMARKS})
    find_package(Catch2 CONFIG REQUIRED)
    add_executable(riesling-bench
        dot.cpp
        grid.cpp
        kernel.cpp
        norm.cpp
        nufft.cpp
        rss.cpp
    )
    set_source_files_properties(
        grid.cpp
        kernel.cpp
        nufft.cpp
        PROPERTIES COMPILE_FLAGS "-ffinite-math-only -funsafe-math-optimizations"
    )
    target_link_libraries(riesling-bench PUBLIC
        rl
        Catch2::Catch2WithMain
    )
    set_target_properties(riesling-bench PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
    )
endif()
