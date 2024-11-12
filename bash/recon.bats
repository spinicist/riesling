@test "Run naive reconstruction" {
    PREFIX="naive"
    MAT=48
    VOX=3
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --nex=0.5 --gradcubes --size=48
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=4
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
    riesling op-nufft --fwd ${PREFIX}-channels.h5 ${PREFIX}-kspace.h5
    riesling recon-lsq ${PREFIX}-kspace.h5 ${PREFIX}-lsq.h5
    # rm ${PREFIX}*.h5
}

@test "Run VCC reconstruction" {
    PREFIX="vcc"
    MAT=48
    VOX=3
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --nex=0.5 --gradcubes --size=48
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=4
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
    riesling op-nufft --fwd ${PREFIX}-channels.h5 ${PREFIX}-kspace.h5
    riesling recon-lsq ${PREFIX}-kspace.h5 ${PREFIX}-lsq.h5 --vcc
    # rm ${PREFIX}*.h5
}

@test "Run frames reconstruction" {
    PREFIX="frames"
    MAT=48
    VOX=3
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --nex=0.5 --gradcubes --size=48
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=4
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
    riesling op-nufft --fwd ${PREFIX}-channels.h5 ${PREFIX}-kspace.h5
    riesling basis-frames --tpf=32 --fpr=2 ${PREFIX}.h5
    riesling recon-lsq ${PREFIX}-kspace.h5 ${PREFIX}-lsq.h5 --basis=${PREFIX}.h5
    # rm ${PREFIX}*.h5
}

@test "SENSE Calib" {
    PREFIX="calib"
    MAT=48
    VOX=3
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --nex=0.5 --gradcubes --size=48
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=4
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
    riesling op-nufft --fwd ${PREFIX}-channels.h5 ${PREFIX}-kspace.h5
    riesling sense-calib ${PREFIX}-kspace.h5 ${PREFIX}-k.h5
    riesling sense-maps ${PREFIX}-k.h5 ${PREFIX}-kspace.h5 ${PREFIX}-maps.h5
    # rm ${PREFIX}*.h5
}