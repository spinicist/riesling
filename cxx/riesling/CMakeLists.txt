add_executable(riesling
    basis/blend.cpp
    basis/bernstein.cpp
    basis/concat.cpp
    basis/echoes.cpp
    basis/fourier.cpp
    basis/img.cpp
    basis/outer.cpp
    basis/sim.cpp
    basis/svd.cpp
    basis/frames.cpp

    data/h5.cpp
    data/compress.cpp
    data/diff.cpp
    data/merge.cpp
    data/nii.cpp
    data/noisify.cpp
    data/slice.cpp

    op/fft.cpp
    op/grad.cpp
    op/grid.cpp
    op/ndft.cpp
    op/nufft.cpp
    op/prox.cpp
    op/sense.cpp
    op/pad.cpp
    op/rss.cpp
    op/wavelets.cpp

    recon/outputs.cpp
    recon/channels.cpp
    recon/lad.cpp
    recon/lsq.cpp
    # recon/pdhg.cpp
    # recon/pdhg-setup.cpp
    recon/rlsq.cpp
    recon/rss.cpp
    # recon/sake.cpp

    sense/calib.cpp
    sense/maps.cpp
    sense/sim.cpp

    util/autofocus.cpp
    util/denoise.cpp
    util/downsamp.cpp
    util/eig.cpp
    util/filter.cpp
    util/phantom.cpp
    util/precon.cpp
    util/psf.cpp
    # util/rovir.cpp
    util/version.cpp

    inputs.cpp
    main.cpp
)
target_link_libraries(riesling PRIVATE
    vineyard
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
if (${BUILD_MONTAGE})
    target_sources(riesling PRIVATE magick.cpp util/montage.cpp)
    target_compile_definitions(riesling PRIVATE -DBUILD_MONTAGE)
    find_package(ImageMagick COMPONENTS Magick++ MagickCore)
    target_link_libraries(riesling PRIVATE ImageMagick::Magick++ ImageMagick::MagickCore)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(riesling PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()
