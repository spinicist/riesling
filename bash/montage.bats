@test "Run basic reconstruction" {
    PREFIX="basic"
    MAT=64
    VOX=3
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --nex=0.5 --gradcubes --size=64
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=4
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
    riesling op-nufft --fwd ${PREFIX}-channels.h5 ${PREFIX}-kspace.h5
    riesling recon-lsq ${PREFIX}-kspace.h5 ${PREFIX}-lsq.h5 --max-its=10
    riesling montage -x ${PREFIX}-lsq.h5 ${PREFIX}.png
    rm *.h5
}