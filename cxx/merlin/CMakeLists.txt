# Option for this one is in main CMakeLists.txt so it is present for the VCPKG call before Project
if(BUILD_MERLIN)
    find_package(args CONFIG REQUIRED)
    find_package(ITK 5.4.0 REQUIRED
    COMPONENTS
      ITKCommon
      ITKImageFilterBase
      ITKRegistrationMethodsv4
      ITKOptimizersv4
    )
    include(${ITK_USE_FILE})
    add_executable(merlin
        main.cpp
        ../args.cpp

        apply.cpp
        merlin.cpp
        register.cpp
        util.cpp
    )
    target_link_libraries(merlin PRIVATE
        rl
        taywee::args
        ${ITK_LIBRARIES}
    )
    target_include_directories(merlin PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    set_target_properties(merlin PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
    )
    install(TARGETS merlin RUNTIME DESTINATION bin)
endif()
