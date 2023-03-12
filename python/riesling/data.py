import h5py
import xarray as xr
import numpy as np

INFO_STRUCTURE = ['matrix', 'tr', 'voxel_size', 'origin', 'direction']
INFO_FORMAT = [('<i8', (3,)), '<f4', ('<f4', (3,)), ('<f4', (3,)), ('<f4', (3,3))]
INFO_DTYPE = np.dtype({'names': INFO_STRUCTURE, 'formats': INFO_FORMAT})

def _create_info(matrix, voxel_size, tr, origin, direction):
    info = np.array([(matrix, tr, voxel_size, origin, direction)], dtype=INFO_DTYPE)
    return info

def _read_info(hdf5_dataset):
    d = {}
    info = np.array(hdf5_dataset, dtype=INFO_DTYPE)[0]
    for key, item in zip(INFO_STRUCTURE, info):
        d[key] = item
    return d

def riesling_read(filename):
    with h5py.File(filename) as f:
        # load meta data, everything that is not the actualy data (e.g. kspace/image) is 
        # considered meta data including trajectory and sampling density correction factors
        meta = {}
        if 'info' in f.keys():
            info = _read_info(f['info'])
            meta.update(info)
        if 'trajectory' in f.keys():
            meta['trajectory'] = np.array(f['trajectory'])
            
        # try to load actual data
        data = None
        dims = []
        data_keys = np.array(['noncartesian','cartesian','channels','image','sense'])
        data_idx = np.in1d(data_keys, np.array([str(f) for f in f.keys()]))
        if data_idx.any():
            key = data_keys[data_idx][0]
            data = f[key]
            if key == 'noncartesian':
                dims = ['channel', 'sample', 'trace', 'slab', 'volume']
            elif key == 'cartesian' or key == 'channels':
                dims = ['channel', 'image', 'x', 'y', 'z']
            elif key == 'sense':
                dims = ['channel', 'x', 'y', 'z']
            elif key == 'image':
                dims = ['image', 'x', 'y', 'z', 'volume']
            else: # this should never happen
                return None 
            dims.reverse() # invert dimension order to match numpy array shape
        else:
            print(f'Could not find any keys matching {data_keys} in file')
            return None
        
        # build xarray output
        if (data.ndim) != len(dims):
            print(f'Number of dimensions in data does not match expected number of dimensions')
            return None
        data = xr.DataArray(data, dims = dims)
        data.attrs.update(meta)
        
        return data

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
