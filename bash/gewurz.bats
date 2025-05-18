@test "Cartesian" {
    PREFIX="c"
    MAT=64
    VOX=4
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --gradcubes --size=96 --cart
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=1
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
    riesling op-nufft --fwd ${PREFIX}-channels.h5 ${PREFIX}-kspace.h5 --osamp=2
}

@test "NUFFT" {
    riesling op-nufft c-kspace.h5 c-nufft.h5
}

@test "GEWURZ" {
    gewurz dft c-kspace.h5 c-dft.h5
}
