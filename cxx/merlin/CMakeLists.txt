option(BUILD_MERLIN "Build the MERLIN binary" ON)
if(${BUILD_BIN})
    find_package(args CONFIG REQUIRED)

    add_executable(merlin
        main.cpp
        args.cpp
        navs.cpp
        move.cpp
    )
    target_link_libraries(merlin PRIVATE
        rl
        taywee::args
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
