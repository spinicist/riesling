option(BUILD_GEWURZ "Build the gewurz binary" OFF)
if(BUILD_GEWURZ)
    enable_language(CUDA)
    find_package(args CONFIG REQUIRED)

    set(CCCL_REPOSITORY "https://github.com/NVIDIA/cccl" CACHE STRING "Git repository to fetch CCCL from")
    set(CCCL_TAG "v2.8.3" CACHE STRING "Git tag/branch to fetch from CCCL repository")
    
    include(${CMAKE_SOURCE_DIR}/cmake/CPM.cmake)
    # This will automatically clone CCCL from GitHub and make the exported cmake targets available
    CPMAddPackage(
        NAME CCCL
        GIT_REPOSITORY "${CCCL_REPOSITORY}"
        GIT_TAG ${CCCL_TAG}
        GIT_SHALLOW ON
        OPTIONS "CCCL_ENABLE_UNSTABLE" ON
    )

    add_executable(gewurz
        main.cpp

        args.cpp
        info.cpp
        sense.cpp
        trajectory.cu
        # test.cu

        cmd/dft.cu
        cmd/precon.cu

        algo/bidiag.cu
        algo/dot.cu
        algo/precon.cu
        
        op/dft.cu
        # op/dft2.cu
        op/recon.cu
        
    )
    set_source_files_properties(
        test.cu
        
        cmd/dft.cu
        cmd/precon.cu
        trajectory.cu
        
        algo/bidiag.cu
        algo/dot.cu
        algo/precon.cu

        op/dft.cu
        op/dft2.cu
        op/recon.cu
        
        PROPERTIES COMPILE_FLAGS "--extended-lambda --expt-relaxed-constexpr"
    )
    target_compile_features(gewurz PUBLIC cuda_std_17)
    target_link_libraries(gewurz PRIVATE
        CCCL::CCCL
        cudax::cudax
        rl
        taywee::args
    )
    target_include_directories(gewurz PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    set_target_properties(gewurz PROPERTIES
        CXX_STANDARD 17
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        CUDA_ARCHITECTURES "75;86"
    )
    install(TARGETS gewurz RUNTIME DESTINATION bin)
endif()
