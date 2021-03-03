#!/bin/bash
#
# Run python reference method reconstruction 

echo "Updating submodule"
git submodule init 
git submodule update

bpath=${PWD}

rrsg_repo=${PWD}/rrsg_challenge_01
rrsg_pyref=${rrsg_repo}/python/rrsg_cgreco/recon.py
config=${rrsg_repo}/python/default.txt

export PYTHONPATH=${PYTHONPATH}:${rrsg_repo}/python

mkdir rrsg_recon
cd rrsg_recon

mkdir challenge_brain
cd challenge_brain


python3 ${rrsg_pyref} --config $config --datafile ${bpath}/rrsg_data/rawdata_brain_radial_96proj_12ch.h5

cd ..
mkdir reference_brain
cd reference_brain 

python3 ${rrsg_pyref} --config $config --datafile ${bpath}/rrsg_data/rawdata_spiral_ETH.h5
