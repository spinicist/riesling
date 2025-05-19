@test "Prep" {
    PREFIX="g"
    MAT=64
    VOX=4
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --gradcubes --size=96 --cart
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=1
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
}

@test "NUFFT->DFT" {
    PREFIX="g"
    riesling op-nufft --fwd ${PREFIX}-channels.h5 ${PREFIX}-ks-nufft.h5
    gewurz dft ${PREFIX}-ks-nufft.h5 ${PREFIX}-img-dft.h5
}

@test "DFT->DFT" {
    PREFIX="g"
    gewurz dft --fwd ${PREFIX}-channels.h5 ${PREFIX}-ks-dft.h5
    gewurz dft ${PREFIX}-ks-dft.h5 ${PREFIX}-img-dft2.h5
}
