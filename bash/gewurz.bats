@test "Cart" {
    MAT=64
    VOX=4
    PREFIX="c"
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --gradcubes --size=96 --cart
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=4
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
    riesling op-nufft --fwd ${PREFIX}-channels.h5 ${PREFIX}-ks-nufft.h5
    gewurz dft ${PREFIX}-ks-nufft.h5 ${PREFIX}-img-dft.h5
    gewurz dft --fwd ${PREFIX}-channels.h5 ${PREFIX}-ks-dft.h5
    gewurz dft ${PREFIX}-ks-dft.h5 ${PREFIX}-img-dft2.h5
}

@test "Radial" {
    MAT=64
    VOX=4
    PREFIX="r"
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --gradcubes --size=96
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=4
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
    riesling op-nufft --fwd ${PREFIX}-channels.h5 ${PREFIX}-ks-nufft.h5
    gewurz dft ${PREFIX}-ks-nufft.h5 ${PREFIX}-img-dft.h5
    gewurz dft --fwd ${PREFIX}-channels.h5 ${PREFIX}-ks-dft.h5
    gewurz dft ${PREFIX}-ks-dft.h5 ${PREFIX}-img-dft2.h5
}
