@test "Run setup" {
    PREFIX="basic"
    MAT=48
    VOX=3
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --gradcubes --size=48
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=4
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
    riesling op-nufft --fwd ${PREFIX}-channels.h5 ${PREFIX}-kspace.h5
}

@test "Run naive reconstruction" {
    riesling recon-lsq basic-kspace.h5 naive-lsq.h5
}

@test "SENSE Calib" {
    PREFIX="calib"
    riesling sense-calib basic-kspace.h5 ${PREFIX}-k.h5
    riesling sense-maps ${PREFIX}-k.h5 basic-kspace.h5 ${PREFIX}-maps.h5
}

@test "Run lowmem reconstruction" {
    riesling recon-lsq basic-kspace.h5 lowmem-lsq.h5 --sense=calib-k.h5 --lowmem
}

@test "Run frames reconstruction" {
    PREFIX="frames"
    riesling basis-frames --tpf=32 --fpr=2 ${PREFIX}.h5
    riesling recon-lsq basic-kspace.h5 ${PREFIX}-lsq.h5 --sense=calib-k.h5 --basis=${PREFIX}.h5
}

@test "Run wavelets reconstruction" {
    PREFIX="waves"
    riesling recon-rlsq basic-kspace.h5 ${PREFIX}-lsq.h5 --sense=calib-k.h5 --wavelets=1e-3
}

@test "Run TV reconstruction" {
    PREFIX="tv"
    riesling recon-rlsq basic-kspace.h5 ${PREFIX}-lsq.h5 --sense=calib-k.h5 --tv=1e-3
}
