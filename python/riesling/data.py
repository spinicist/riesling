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

def write(filename, data, data_type='noncartesian', compression='gzip'):
    # match data dimensions and data format based on data_type
    if data_type == 'noncartesian':
        data_dims = ['channel', 'sample', 'trace', 'slab', 'volume']
        dtype = 'c8'
    elif data_type == 'cartesian' or data_type == 'channels':
        data_dims = ['channel', 'image', 'x', 'y', 'z', 'volume']
        dtype = 'c8'
    elif data_type == 'sense':
        data_dims = ['channel', 'x', 'y', 'z']
        dtype = 'c8'
    elif data_type == 'image':
        data_dims = ['image', 'x', 'y', 'z', 'volume']
        dtype = 'c8'
    elif data_type == 'sdc':
        data_dims = ['sample', 'trace']
        dtype = 'f8'
    elif data_type == 'basis':
        data_dims = ['trace', 'volume']
        dtype = 'f8'
    else:
        AssertionError(f'Unknown data type {data_type}')

    if not isinstance(data, xr.DataArray):
        data = xr.DataArray(data, attrs={}, dims=data_dims) # make it a DataArray with empty header and just assume that dimensions match

    # check for info header information
    if data_type != 'sdc' and data_type != 'basis': # sdc and basis do not require an info header
        if not 'matrix' in data.attrs:
            AssertionError('Data object must contain "matrix" attribute')
        if not 'voxel_size' in data.attrs:
            AssertionError('Data object must contain "voxel_size" attribute')
        if not 'tr' in data.attrs:
            AssertionError('Data object must contain "tr" attribute')
        if not 'origin' in data.attrs:
            AssertionError('Data object must contain "origin" attribute')
        if not 'direction' in data.attrs:
            AssertionError('Data object must contain "direction" attribute')

    # check if additional information for the data types are present in th header
    requires_traj = (data_type == 'noncartesian')
    if requires_traj:
        if not 'trajectory' in data.attrs:
            AssertionError('Noncartesian data object must contain trajectory')

        if data.attrs['trajectory'].ndim != 3:
            AssertionError('Trajectory must be 3 dimensional (co-ords, samples, traces)')
        if data.attrs['trajectory'].shape[2] > 3:
            AssertionError('Trajectory cannot have more than 3 co-ordinates')
        if np.max (data.attrs['trajectory']) > 0.5:
            AssertionError('Trajectory cotained co-ordinates greater than 0.5')

    with h5py.File(filename, 'w') as out_f:
        # write info header
        if {'matrix','voxel_size','tr','origin','direction'} <= data.attrs.keys():
            out_f.create_dataset('info', data=_create_info(
                data.attrs['matrix'],
                data.attrs['voxel_size'],
                data.attrs['tr'],
                data.attrs['origin'],
                data.attrs['direction'])
            )

        # write additional information
        if 'trajectory' in data.attrs:
            out_f.create_dataset('trajectory', data=data.attrs['trajectory'], chunks=np.shape(data.attrs['trajectory']), compression=compression)
        # transpose data to right dimensions
        data_dims.reverse() # invert dimension order to match numpy array shape
        data = data.copy() # deep copy
        data = data.transpose(*data_dims)

        # write data
        out_f.create_dataset(data_type, dtype=dtype, data=data.data, chunks=np.shape(data.data), compression=compression)
        out_f.close()


def read(filename):
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
        data_keys = np.array(['noncartesian','cartesian','channels','image','sense','sdc','basis'])
        data_idx = np.in1d(data_keys, np.array([str(f) for f in f.keys()]))
        if data_idx.any():
            key = data_keys[data_idx][0]
            data = f[key]
            if key == 'noncartesian':
                dims = ['channel', 'sample', 'trace', 'slab', 'volume']
            elif key == 'cartesian' or key == 'channels':
                dims = ['channel', 'image', 'x', 'y', 'z', 'volume']
            elif key == 'sense':
                dims = ['channel', 'image', 'x', 'y', 'z']
            elif key == 'image':
                dims = ['image', 'x', 'y', 'z', 'volume']
            elif key == 'sdc':
                dims = ['sample', 'trace']
            elif key == 'basis':
                dims = ['trace', 'volume']
            else: # this should never happen
                return None 
            dims.reverse() # invert dimension order to match numpy array shape
        else:
            print(f'Could not find any keys matching {data_keys} in file')
            return None
        
        # build xarray output
        if (data.ndim) != len(dims):
            print(f'Number of dimensions in data does not match expected number of dimensions')
            print(f'Data has shape {data.shape} and detected dimensions are {dims}')
            return None
        data = xr.DataArray(data, dims = dims)
        data.attrs.update(meta)
        
        if len(data.name) > 0 and data.name[0] == '/': # name can for unknown reason start with a /
            data.name = data.name[1:] # remove / at beginning

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
