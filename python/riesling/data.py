import h5py
import numpy as np

def _create_info(matrix, voxel_size, tr, origin, direction):
    D = np.dtype({'names': ['matrix', 'tr', 'voxel_size', 'origin', 'direction'],
                  'formats': [('<i8', (3,)), '<f4', ('<f4', (3,)), ('<f4', (3,)), ('<f4', (3,3))]})
    info = np.array([(matrix, tr, voxel_size, origin, direction)], dtype=D)
    return info

def write_noncartesian(fname, kspace, traj, matrix, voxel_size=[1,1,1], tr=1, origin=[0,0,0], direction=np.eye(3)):
    if kspace.ndim != 5:
        raise('K-space must be 5 dimensional (channels, samples, traces, slabs, volumes)')
    if traj.ndim != 3:
        raise('Trajectory must be 3 dimensional (co-ords, samples, traces)')
    if traj.shape[2] > 3:
        raise('Trajectory cannot have more than 3 co-ordinates')
    if np.max(np.abs(traj)) > 0.5:
        raise('Trajectory cotained co-ordinates greater than 0.5')

    # Create new H5 file
    with h5py.File(fname, 'w') as out_f:
        out_f.create_dataset("info", data=_create_info(matrix, voxel_size, tr, origin, direction))
        out_f.create_dataset('trajectory', data=traj, chunks=np.shape(traj), compression="gzip")
        out_f.create_dataset("noncartesian", dtype='c8', data=kspace, chunks=np.shape(kspace), compression="gzip")
        out_f.close()
