####################################################
#
# Convert the CG SENSE Reproducibility challenge brain
# dataset to riesling h5 format.
#
# How to use:
# - Download brain challenge and reference dataset from
#        https://zenodo.org/record/3975887#.X9EE6l7gokg
# - Put data into folder named rrsg_data
#
# - Create folder for riesling data named riesling_data
# Run script
#   python3 convert_data.py
#
# Emil Ljungberg and Tobias Wood
# December 2020, King's Colleg London
####################################################

import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys


def create_info(matrix, voxel_size, read_points, read_gap, spokes_hi, spokes_lo, lo_scale,
                channels, volumes, tr, origin, direction):
    D = np.dtype({'names': [
        'matrix', 
        'voxel_size', 
        'read_points', 
        'read_gap', 
        'spokes_hi', 
        'spokes_lo', 
        'lo_scale',
        'channels', 
        'volumes', 
        'tr', 
        'origin', 
        'direction'],
        'formats': [
        ('<i8', (3,)), 
        ('<f4', (3,)), 
        '<i8', 
        '<i8', 
        '<i8', 
        '<i8', 
        '<f4',
        '<i8', 
        '<i8', 
        '<f4', 
        ('<f4', (3,)), 
        ('<f4', (9,))]
        # 'offsets': [0, 24, 40, 48, 56, 64, 72, 80, 88, 96, 100, 112],
        # 'itemsize': 170
        })

    info = np.array([(matrix, voxel_size, read_points, read_gap, spokes_hi, spokes_lo, lo_scale,
                      channels, volumes, tr, origin, direction)], dtype=D)

    return info


def convert_data(input_fname, output_fname, matrix, voxel_size):
    data_f = h5py.File(input_fname, 'r')
    rawdata = data_f['rawdata'][...]
    traj = data_f['trajectory'][...]
    data_f.close()

    # Scale trajectory
    traj = traj/np.max(abs(traj))
    [nd, npoints, nshots] = np.shape(traj)

    # If 2D trajectory extend to third dimension
    if nd == 2:
        traj2 = np.zeros((3, npoints, nshots))
        traj2[0:2, :, :] = traj
        traj = traj2

    # Create info struct
    read_points = np.shape(rawdata)[1]
    read_gap = 0
    spokes_hi = np.shape(rawdata)[2]
    spokes_lo = 0
    channels = np.shape(rawdata)[3]
    volumes = 1
    tr = 1
    lo_scale = 8
    origin = [0, 0, 0]
    direction = [1, 0, 0, 0, 1, 0, 0, 0, 1]
    h5_info = create_info(matrix, voxel_size, read_points, read_gap, spokes_hi, spokes_lo, lo_scale,
                          channels, volumes, tr, origin, direction)

    # Reshape data
    rawdata_rs = np.transpose(rawdata[0, :, :, :], [1, 0, 2])
    traj_rs = np.transpose(traj, [2, 1, 0])

    # Create new H5 file
    out_f = h5py.File(output_fname, 'w')
    out_f.create_dataset("info", data=h5_info)
    h5_traj = np.reshape(np.reshape(traj_rs, (1, np.prod(traj_rs.shape))), [
        traj_rs.shape[2], traj_rs.shape[1], traj_rs.shape[0]])

    out_f.create_dataset('traj', data=h5_traj,
                         chunks=np.shape(h5_traj), compression="gzip")

    grp = out_f.create_group("data")
    h5_data = np.reshape(np.reshape(rawdata_rs, (1, np.prod(rawdata_rs.shape))), [
        rawdata_rs.shape[2], rawdata_rs.shape[1], rawdata_rs.shape[0]])

    grp.create_dataset("0000", dtype='c8', data=h5_data,
                       chunks=np.shape(h5_data), compression="gzip")
    out_f.close()

    print("H5 file saved to {}".format(output_fname))


def main():
    # Challenge data
    f = 'rawdata_brain_radial_96proj_12ch.h5'
    convert_data(input_fname='rrsg_data/%s' % f,
                 output_fname='riesling_data/riesling_%s' % f,
                 matrix=[256, 256, 1], voxel_size=[0.78, 0.78, 2])

    # Reference data
    f = 'rawdata_spiral_ETH.h5'
    convert_data(input_fname='rrsg_data/%s' % f,
                 output_fname='riesling_data/riesling_%s' % f,
                 matrix=[220, 220, 1], voxel_size=[1, 1, 2])

if __name__ == "__main__":
    main()
