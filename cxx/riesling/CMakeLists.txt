option(BUILD_BIN "Build the binary" ON)
if(BUILD_BIN)
    find_package(args CONFIG REQUIRED)

    add_executable(riesling
        main.cpp
        ../args.cpp
        inputs.cpp
        regularizers.cpp

        basis/blend.cpp
        basis/bernstein.cpp
        basis/concat.cpp
        basis/decay.cpp
        basis/echoes.cpp
        basis/fourier.cpp
        basis/img.cpp
        basis/navs.cpp
        basis/outer.cpp
        basis/sim.cpp
        basis/slice.cpp
        basis/svd.cpp
        basis/frames.cpp

        data/h5.cpp
        data/compress.cpp
        data/diff.cpp
        data/merge.cpp
        data/move.cpp
        data/norm.cpp
        data/nii.cpp
        data/noisify.cpp
        data/real.cpp
        data/slicenc.cpp
        data/sliceimg.cpp

        op/fft.cpp
        op/grad.cpp
        op/grid.cpp
        # op/ndft.cpp
        op/nufft.cpp
        op/prox.cpp
        op/sense.cpp
        op/pad.cpp
        op/rss.cpp
        op/wavelets.cpp

        recon/lsq.cpp
        recon/rlsq.cpp
        recon/rss.cpp
        # recon/sake.cpp

        sense/calib.cpp
        sense/maps.cpp
        sense/sim.cpp

        # util/autofocus.cpp
        util/denoise.cpp
        util/downsamp.cpp
        util/eig.cpp
        util/log.cpp
        util/filter.cpp
        util/mask.cpp        
        util/phantom.cpp
        util/precon.cpp
        util/psf.cpp
        util/resamp.cpp
        util/rovir.cpp
        util/version.cpp
    )
    target_link_libraries(riesling PRIVATE
        rl
        taywee::args
    )
    target_include_directories(riesling PRIVATE ${CMAKE_CURRENT_LIST_DIR})
    set_target_properties(riesling PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
        CXX_EXTENSIONS OFF
        CXX_VISIBILITY_PRESET hidden
    )
    install(TARGETS riesling RUNTIME DESTINATION bin)

    option(BUILD_MONTAGE "Build the montage command. Requires ImageMagick" OFF)
    if (BUILD_MONTAGE)
        target_sources(riesling PRIVATE magick.cpp util/montage.cpp)
        target_compile_definitions(riesling PRIVATE -DBUILD_MONTAGE)
        find_package(ImageMagick COMPONENTS Magick++ MagickCore)
        target_link_libraries(riesling PRIVATE ImageMagick::Magick++ ImageMagick::MagickCore)
        find_package(OpenMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(riesling PRIVATE OpenMP::OpenMP_CXX)
        endif()
    endif()
endif()
