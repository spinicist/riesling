# Example - CG-SENSE Challenge

Scripts to test `riesling` on the ISMRM CG-SENSE Reproducibility challenge. (_CG-SENSE revisited: Results from the first ISMRM reproducibility challenge, [10.1002/mrm.28569](https://onlinelibrary.wiley.com/doi/10.1002/mrm.28569)_)

## 1. Download data
You can automatically download the data from Zenodo using
```sh
bash download_data.sh
```

Or download it manually and place it in a subfolder called `rrsg_data`. Download link: https://zenodo.org/record/3975887#.X9EE6l7gokg


## 2. Convert to riesling format
To convert the data to `riesling` .h5 format, create a `rielsing_data` folder and perform converstion using
```sh
mkdir riesling_data
python3 convert_data.py
```

The conversion script is also a useful reference to understand the `riesling` data format.

## 3. Run riesling recon
To perform CG-SENSE recon with `riesling` run
```sh
bash riesling_recon.sh
```

## 4. Reference method
The reference method we compare to is the Python implementation from the challenge. This is added as a submodule in this repository. To run the comparison recon
```sh
bash rrsg_recon.sh
```

## 5. Comparison images
To produce comparison images run
```sh
python3 riesling_rrsg_compare.py
```