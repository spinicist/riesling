#!/bin/bash
#
# Run python reference method reconstruction 
# Requires the RRSG_CGSENSE repository from:
#   https://github.com/ISMRM/rrsg_challenge_01

rrsg_repo=<insert_your_path>
rrsg_pyref=${rrsg_repo}/python/rrsg_cgreco/recon.py
config=${rrsg_repo}/python/default.txt

mkdir rrsg_recon
cd rrsg_recon

mkdir challenge_brain
cd challenge_brain

python3 $rrsg_pyref --config $config --datafile rrsg_data/rawdata_brain_radial_96proj_12ch.h5

cd ..
mkdir reference_brain
cd reference_brain 

python3 $rrsg_pyref --config $config --datafile ${p}/rawdata_spiral_ETH.h5
