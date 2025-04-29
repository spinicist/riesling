option(BUILD_GEWURZ "Build the gewurz binary" OFF)
if(BUILD_GEWURZ)
    enable_language(CUDA)
    find_package(args CONFIG REQUIRED)
    find_package(Thrust REQUIRED CONFIG)
    thrust_create_target(Thrust DEVICE CUDA)
    add_executable(gewurz
        main.cpp
        ../args.cpp
        inputs.cpp
        # outputs.cpp
        dft.cu
    )
    set_source_files_properties(
        dft.cu
        PROPERTIES COMPILE_FLAGS --extended-lambda
    )
    target_link_libraries(gewurz PRIVATE
        Thrust
        rl
        taywee::args
    )
    target_include_directories(gewurz PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    set_target_properties(gewurz PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        CUDA_ARCHITECTURES "75"
    )
    install(TARGETS gewurz RUNTIME DESTINATION bin)
endif()
