# Example - CG-SENSE Challenge

Scripts to test `riesling` on the ISMRM CG-SENSE Reproducibility challenge. (_CG-SENSE revisited: Results from the first ISMRM reproducibility challenge, [10.1002/mrm.28569](https://onlinelibrary.wiley.com/doi/10.1002/mrm.28569)_)

## 1. Download data
The data from the challenge and place in a subfolder called `rrsg_data`. Download link: https://zenodo.org/record/3975887#.X9EE6l7gokg

## 2. Convert to riesling format
To convert the data to `riesling` .h5 format, run
```sh
python3 convert_data.py
```

The conversion script is also a useful reference to understand the `riesling` data format.

## 3. Run riesling recon
Example of how to run CG-SENSE recon is in `riesling_recon.sh`

## 4. Reference method
As a comparison, download the reference python method by cloning their [github repository](https://github.com/ISMRM/rrsg_challenge_01). To run the reference recon look at `rrsg_recon.sh` for a demo.

## 5. Comparison images
To produce comparison images run
```sh
python3 riesling_rrsg_compare.py
```