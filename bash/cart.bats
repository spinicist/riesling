@test "Run setup" {
    PREFIX="cart"
    MAT=48
    VOX=3
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --gradcubes --size=48 --cart
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=4
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
    riesling op-nufft --fwd ${PREFIX}-channels.h5 ${PREFIX}-kspace.h5
}

@test "Run naive reconstruction" {
    PREFIX="cart"
    riesling recon-lsq ${PREFIX}-kspace.h5 ${PREFIX}-lsq.h5 --osamp=1 --tophat
}
