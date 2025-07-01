@test "Grad (Requires Manual Inspection of Output)" {
    PREFIX="grad"
    MAT=48
    VOX=3
    riesling phantom ${PREFIX}-phantom.h5 --matrix=$MAT --vox-size=$VOX --gradcubes --size=48

    riesling op-grad ${PREFIX}-phantom.h5 ${PREFIX}-grad.h5 --fwd
    riesling op-grad ${PREFIX}-grad.h5 ${PREFIX}-div.h5 --fwd --vec
    riesling op-grad ${PREFIX}-div.h5 ${PREFIX}-grad2.h5 --vec
    riesling op-grad ${PREFIX}-grad2.h5 ${PREFIX}-phantom2.h5
}
