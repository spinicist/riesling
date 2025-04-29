option(BUILD_FURMINT "Build the furmint binary" OFF)
if(BUILD_FURMINT)
    set(CUTENSOR_ROOT ${CMAKE_CURRENT_LIST_DIR} CACHE PATH "Path to cutensor")
    enable_language(CUDA)
    find_package(args CONFIG REQUIRED)
    find_package(CUDA 11.1 REQUIRED)
    include_directories("${CUDA_INCLUDE_DIRS}")
    
    add_library(cutensor SHARED IMPORTED)
    set(CUTENSOR_LIBRARY_NAME "libcutensor.so")
    set(LIB_DIR "lib/${CUDA_VERSION_MAJOR}")
    set_target_properties(cutensor PROPERTIES
        IMPORTED_LOCATION "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_NAME}"
        IMPORTED_IMPLIB "${CUTENSOR_ROOT}/${LIB_DIR}/${CUTENSOR_LIBRARY_DEF}"
        INTERFACE_INCLUDE_DIRECTORIES "${CUTENSOR_ROOT}/include")
    

    add_executable(furmint
        main.cpp
        ../args.cpp
        inputs.cpp
        # outputs.cpp
        dft.cu
    )
    target_link_libraries(furmint PRIVATE
        cutensor
        cudart
        rl
        taywee::args
    )
    target_include_directories(furmint PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    set_target_properties(furmint PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
        CUDA_ARCHITECTURES "75"
    )
    install(TARGETS furmint RUNTIME DESTINATION bin)
endif()
