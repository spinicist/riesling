@test "Denoise" {
    MAT=64
    VOX=2
    ITS=16
    riesling phantom phantom.h5 --matrix=$MAT --vox-size=$VOX --gradcubes --size=56
    riesling noisify phantom.h5 phantom-noisy.h5 --std=0.1
    riesling denoise phantom-noisy.h5 phantom-tv.h5 --tv=0.1  --max-its=${ITS}
    riesling denoise phantom-noisy.h5 phantom-lad.h5 --tv=0.2  --max-its=${ITS} --lad
    riesling denoise phantom-noisy.h5 phantom-tv2.h5 --tv2=0.1 --max-its=${ITS}
    riesling denoise phantom-noisy.h5 phantom-tgv.h5 --tgv=0.1 --max-its=${ITS}
    riesling denoise phantom-noisy.h5 phantom-wav.h5 --wavelets=0.1 --max-its=${ITS}
}
