@test "Convert 5D to nii" {
    PREFIX="nii"
    MAT=48
    VOX=3
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --nex=0.5 --gradcubes --size=48
    riesling nii ${PREFIX}-phantom.h5 ${PREFIX}-phantom.nii
    rm ${PREFIX}*.h5 ${PREFIX}*.nii
}

@test "Convert 6D to nii" {
    PREFIX="nii"
    MAT=48
    VOX=3
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --nex=0.5 --gradcubes --size=48
    riesling sense-sim ${PREFIX}-sim-sense.h5 --matrix=$MAT --vox-size=$VOX --channels=4
    riesling op-sense --fwd ${PREFIX}-phantom.h5 ${PREFIX}-channels.h5 ${PREFIX}-sim-sense.h5
    riesling nii ${PREFIX}-channels.h5 ${PREFIX}-channels.nii
    rm ${PREFIX}*.h5 ${PREFIX}*.nii
}
