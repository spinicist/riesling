import h5py
import xarray as xr
import numpy as np

INFO_FIELDS = ['voxel_size', 'origin', 'direction', 'tr']
INFO_FORMAT = [('<f4', (3,)), ('<f4', (3,)), ('<f4', (3, 3)), '<f4']
INFO_DTYPE = np.dtype({'names': INFO_FIELDS, 'formats': INFO_FORMAT})

def write(filename, data=None, trajectory=None, matrix=None, info=None, meta=None, compression='gzip'):
    with h5py.File(filename, 'w') as out_f:
        # create dataset and assign dimension labels
        out_f.create_dataset('data', dtype='c8', data=data.data, chunks=np.shape(data.data), compression=compression)
        for idd, dim in enumerate(data.dims):
            out_f['data'].dims[idd].label = dim

        # add trajectory property
        if trajectory is not None:
            assert trajectory.ndim == 3, 'Trajectory must be 3 dimensional (co-ords, samples, traces)'
            assert trajectory.shape[2] <= 3, 'Trajectory cannot have more than 3 co-ordinates'
            traj = out_f.create_dataset('trajectory', dtype='f4', data=trajectory, compression=compression)
            traj.dims[2].label = "k"
            traj.dims[1].label = "sample"
            traj.dims[0].label = "trace"
            if matrix is not None:
                traj.attrs.create('matrix', np.array(matrix), dtype=h5py.h5t.array_create(h5py.h5t.NATIVE_INT64, (3,)))

        # add info struct
        if info is not None:
            out_f.create_dataset('info', data=np.array([tuple([info[f] for f in INFO_FIELDS])], dtype=INFO_DTYPE))

        # add meta info
        if meta is not None:
            meta_g = out_f.create_group('meta')
            for k, v in meta:
                meta_g.create_dataset(k, data=v)

def read_data(filename, dset=None):
    with h5py.File(filename) as f:
        if dset is None:
            dset = 'data'
        return xr.DataArray(f[dset], dims=[d.label for d in f[dset].dims])

def read_info(filename):
    with h5py.File(filename) as f:
        if 'info' in f.keys():
            info_dset = np.array(f['info'], dtype=INFO_DTYPE)[0]
            info_dict = {}
            for key, item in zip(INFO_FIELDS, info_dset):
                info_dict[key] = item
            return info_dict
        else:
            return None

def read_trajectory(filename):
    with h5py.File(filename) as f:
        if 'trajectory' in f.keys():
            return np.array(f['trajectory'])

def read_meta(filename):
    with h5py.File(filename) as f:
        if 'meta' in f.keys():
            meta = {}
            for k in f['meta'].keys():
                meta[k] = f['meta'][k][0]
            return meta
        else:
            return None
