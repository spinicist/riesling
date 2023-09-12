import os
import sys
import inspect
import riesling
import unittest
import xarray as xr
import numpy as np

# define test order to make reusing of previous results possible
order = ['test_image', 'test_sense', 'test_kspace', 'test_rss', 'test_nufft', 'test_sdc']
def compare(a, b):
    if order.index(a) < order.index(b):
        return -1
    elif order.index(a) < order.index(b):
        return 1
    else:
        return 0
unittest.defaultTestLoader.sortTestMethodsUsing = compare

# helper class to plot results if run with -vv
def plot(data, dims, sel, title=''):
    if '-vv' in sys.argv:
        import matplotlib.pyplot as plt

        data = data.copy()
        data = np.abs(data)
        data = data.isel(sel) # select subset
        if data.ndim > len(dims): # transpose displayed dimensions to beginning
            data = data.transpose(*dims, ...)
        else:                
            data = data.transpose(*dims)
        if data.ndim > 2: # linearize all other dimensions
            lin_dims = list(reversed(data.dims[1:]))
            data = data.stack({f"{str(lin_dims)}": lin_dims})
        
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(data)
        ax.set_title(title)
        ax.set_xlabel(data.dims[1])
        ax.set_ylabel(data.dims[0])
        plt.show()

class TestReadWrite(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.prefix = 'unittest'
        cls.matrix=64
        cls.voxsz=3
        cls.channels=4
    
    @classmethod
    def tearDownClass(cls):
        os.system(f'rm {cls.prefix}*.h5')

    def test_image(self):
        os.system(f'../build/riesling phantom {self.prefix} --matrix={self.matrix} --vox-size={self.voxsz} --nex=0.5 --gradcubes --size=64')
        data = riesling.data.read(f'{self.prefix}.h5')
        self.assertEqual(data.shape, (1, 64, 64, 64, 1))
        self.assertEqual(data.dims, ('volume', 'z', 'y', 'x', 'image'))
        plot(data, ['x','y'], {'volume':0, 'image':0, 'z':[16,32,48]}, title='phantom')

        riesling.data.write(f'{self.prefix}_rewrite.h5', data, data_type='image')
        data_reload = riesling.data.read(f'{self.prefix}_rewrite.h5')
        np.testing.assert_almost_equal(data.data, data_reload.data)        
        np.testing.assert_almost_equal(data.attrs['trajectory'], data_reload.attrs['trajectory'])
        self.assertEqual(data.dims, data_reload.dims)

    def test_sense(self):
        os.system(f'../build/riesling sense-sim {self.prefix}-sim --matrix={self.matrix} --vox-size={self.voxsz} --channels={self.channels}')
        data = riesling.data.read(f'{self.prefix}-sim-sense.h5')
        self.assertEqual(data.shape, (64, 64, 64, 4))
        self.assertEqual(data.dims, ('z', 'y', 'x', 'channel'))
        plot(data, ['x','y'], {'z':32}, title='sim-sense')

        riesling.data.write(f'{self.prefix}-sim-sense_rewrite.h5', data, data_type='sense')
        data_reload = riesling.data.read(f'{self.prefix}-sim-sense_rewrite.h5')
        np.testing.assert_almost_equal(data.data, data_reload.data)
        self.assertEqual(data.dims, data_reload.dims)

    def test_kspace(self):
        os.system(f'../build/riesling recon --fwd --sense={self.prefix}-sim-sense.h5 --sense-fov=-1,-1,-1 {self.prefix}.h5')
        data = riesling.data.read(f'{self.prefix}-recon.h5')
        self.assertEqual(data.shape, (1, 1, 2048, 64, 4))
        self.assertEqual(data.dims, ('volume', 'slab', 'trace', 'sample', 'channel'))
        self.assertEqual(data.attrs['trajectory'].shape, (2048, 64, 3))
        plot(data, ['sample', 'trace'], {'volume':0, 'slab':0, 'channel':0, 'trace': range(0, 256)}, title='recon')

        riesling.data.write(f'{self.prefix}-recon_rewrite.h5', data, data_type='noncartesian')
        data_reload = riesling.data.read(f'{self.prefix}-recon_rewrite.h5')
        np.testing.assert_almost_equal(data.data, data_reload.data)
        np.testing.assert_almost_equal(data.attrs['trajectory'], data_reload.attrs['trajectory'])
        self.assertEqual(data.dims, data_reload.dims)

    def test_rss(self):
        os.system(f'../build/riesling rss {self.prefix}-recon.h5')
        data = riesling.data.read(f'{self.prefix}-recon-rss.h5')
        self.assertEqual(data.shape, (1, 64, 64, 64, 1))
        self.assertEqual(data.dims, ('volume', 'z', 'y', 'x', 'image'))
        plot(data, ['x','y'], {'volume':0, 'image':0, 'z':[16,32,48]}, title='rss')

        riesling.data.write(f'{self.prefix}-recon-rss_rewrite.h5', data, data_type='image')
        data_reload = riesling.data.read(f'{self.prefix}-recon-rss_rewrite.h5')
        np.testing.assert_almost_equal(data.data, data_reload.data)
        self.assertEqual(data.dims, data_reload.dims)

    def test_nufft(self):
        os.system(f'../build/riesling nufft {self.prefix}-recon.h5')
        data = riesling.data.read(f'{self.prefix}-recon-nufft.h5')
        self.assertEqual(data.shape, (1, 64, 64, 64, 1, 4))
        self.assertEqual(data.dims, ('volume', 'z', 'y', 'x', 'image', 'channel'))
        plot(data, ['x','y'], {'volume':0, 'image':0, 'z':[16,32,48], 'channel':0}, title='nufft - channel=0')

        riesling.data.write(f'{self.prefix}-recon-nufft_rewrite.h5', data, data_type='channels')
        data_reload = riesling.data.read(f'{self.prefix}-recon-nufft_rewrite.h5')
        np.testing.assert_almost_equal(data.data, data_reload.data)
        self.assertEqual(data.dims, data_reload.dims)

    def test_sdc(self):
        data = riesling.data.read(f'{self.prefix}-recon.h5')
        sdc = xr.DataArray(np.ones(data.attrs['trajectory'].shape[:-1]), dims=('trace','sample')) # build dummy weights array filled with ones
        riesling.data.write(f'{self.prefix}-sdc.h5', sdc, data_type='sdc')
        sdc_reload = riesling.data.read(f'{self.prefix}-sdc.h5')
        np.testing.assert_almost_equal(sdc.data, sdc_reload.data)

        # run nufft again to check if it works with the custom sdc
        os.system(f'../build/riesling nufft {self.prefix}-recon.h5 --sdc={self.prefix}-sdc.h5 -o {self.prefix}-recon-sdc')
        data = riesling.data.read(f'{self.prefix}-recon-sdc-nufft.h5')
        self.assertEqual(data.shape, (1, 64, 64, 64, 1, 4))
        self.assertEqual(data.dims, ('volume', 'z', 'y', 'x', 'image', 'channel'))
        plot(data, ['x','y'], {'volume':0, 'image':0, 'z':[16,32,48], 'channel':0}, title='nufft - channel=0')

if __name__ == "__main__":
    unittest.main()