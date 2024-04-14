add_executable(riesling
    autofocus.cpp
    basis.cpp
    basis-fourier.cpp
    basis-img.cpp
    basis-sim.cpp
    blend.cpp
    compress.cpp
    data.cpp
    denoise.cpp
    downsamp.cpp
    eig.cpp
    fft.cpp
    filter.cpp
    frames.cpp
    grad.cpp
    grid.cpp
    h5.cpp
    ipop-basis.cpp
    ipop-combine.cpp
    # lad.cpp
    lookup.cpp
    merge.cpp
    nii.cpp
    noisify.cpp
    ndft.cpp
    nufft.cpp
    op.cpp
    op-sense.cpp
    pad.cpp
    pdhg.cpp
    pdhg-setup.cpp
    phantom.cpp
    precond.cpp
    prox.cpp
    psf.cpp
    recon.cpp
    recon-lsq.cpp
    recon-rlsq.cpp
    recon-rss.cpp
    recon-sense.cpp
    rss.cpp
    sdc.cpp
    sense.cpp
    sense-calib.cpp
    sense-sim.cpp
    slice.cpp
    util.cpp
    version.cpp
    wavelets.cpp
    zinfandel.cpp
    main.cpp
)
target_link_libraries(riesling PRIVATE
    vineyard
)
set_target_properties(riesling PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    CXX_VISIBILITY_PRESET hidden
)
install(TARGETS riesling RUNTIME DESTINATION bin)

option(BUILD_MONTAGE "Build the montage command. Requires ImageMagick" OFF)

if (${BUILD_MONTAGE})
    target_sources(riesling PRIVATE montage.cpp)
    target_compile_definitions(riesling PRIVATE -DBUILD_MONTAGE)
    execute_process(COMMAND "Magick++-config" "--cxxflags" OUTPUT_VARIABLE IM_CXX)
    execute_process(COMMAND "Magick++-config" "--ldflags" OUTPUT_VARIABLE IM_LD)
    string(STRIP ${IM_CXX} IM_CXX)
    string(STRIP ${IM_LD} IM_LD)
    separate_arguments(IM_CXX)
    separate_arguments(IM_LD)
    target_compile_options(riesling PRIVATE ${IM_CXX})
    target_link_options(riesling PRIVATE ${IM_LD})
endif()
