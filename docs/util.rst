Utilities
=========

RIESLING includes a number of utility commands.

* `h5`_
* `nii`_
* `phantom`_
* `precond`_
* `psf`_
* `compress`_
* `rovir`_
* `downsamp`_
* `slice`_

h5
---

Prints out information from/about a RIESLING ``.h5`` file. By default lists all datasets and their dimensions.

*Usage*

.. code-block:: bash

    riesling h5 file.h5 --info

*Important Options*

* ``--dim=D``, ``--dset=DS``

    Prints a specific dimension size from the specified dataset to stdout.

* ``--meta=M``, ``--all``

    Prints a specific key or all key-value pairs from the meta-data to stdout.

nii
---

*Usage*

.. code-block:: bash
    
    riesling nii file.h5 output_file.nii

*Output*

A NIfTI image containing the desired images as separate volumes. The ``.nii`` extension will be added automatically.

*Important Options*

* ``--mag``

    Output magnitude images instead of complex.

* ``--dset=NAME``

    Specify the dataset you want to convert. The default is ``image``.

phantom
-------

Create a phantom image.

*Usage*

.. code-block:: bash
    
    riesling phantom phantom.h5 --matrix=64

*Output*

The specified file (above ``phantom.h5``) containing the phantom image, trajectory and k-space data.

*Important Options*

* ``--matrix=M``

    The matrix size (isotropic matrix assumed).

* ``--gradcubes``

    The phantom will be cubes with gradients along the different dimensions instead of the default Shepp-Logan phantom.

psf
---

Calculates the Point Spread Function by solving the NUFFT for a dataset of ones.

*Usage*

.. code-block:: bash
    
    riesling psf input.h5 psf.h5

*Important Options*

* ``--mtf``

    Also save the Modulation Transfer Function (Fourier Transform of the PSF)


precond
-------

Calculate the preconditioner for a particular trajectory up-front. The single-channel preconditioner implemented in ``riesling`` is a property only of the trajectory and hence can be re-used between reconstructions.

*Usage*

.. code-block:: bash

    riesling precond file.h5 output.h5

*Output*

``output.h5`` containing the preconditioner.

*Important Options*

* ```--pre-bias=N``

    In a sub-space reconstruction it is possible for the preconditioner calculation to contain divide-by-zero problems. This option adds a bias to the calculation to prevent this causing problems. The default value is 1.

compress
--------

Reduce the channel count using PCA coil compression. See `Huang, F., Vijayakumar, S., Li, Y., Hertel, S. & Duensing, G. R. A software channel compression technique for faster reconstruction with many channels. Magnetic Resonance Imaging 26, 133–141 (2008). <https://linkinghub.elsevier.com/retrieve/pii/S0730725X07002731>`_.

*Usage*

.. code-block:: bash

    riesling compress file.h5 compressed.h5

*Important Options*

* ``--save=file.h5``, ``--cc-file=file.h5``

    Save the compression matrix to a file to re-use on other files.

* ``--channels=N``

    Compress to N channels.

* ``--energy=E``

    Retain the number of channels required to retain the specified fraction of the variance/energy. Valid values are between 0 and 1.

* ``--pca-samp=ST,SZ``

    Take the samples for PCA from `ST` to `ST + SZ` along the read direction.

* ``--pca-traces=ST,SZ,STRIDE``

    Take the samples for PCA from `ST` to `ST + SZ` every `STRIDE` along the spoke direction.

rovir
--------

Calculate a compression matrix using the ROVir method. See `D. Kim, S. F. Cauley, K. S. Nayak, R. M. Leahy, and J. P. Haldar, ‘Region‐optimized virtual (ROVir) coils: Localization and/or suppression of spatial regions using sensor‐domain beamforming’, Magn. Reson. Med., vol. 86, no. 1, pp. 197–212, Jul. 2021 <https://onlinelibrary.wiley.com/doi/10.1002/mrm.28706>`_.

The input to ROVir is a set of channel images, you can make these using ``op-nufft``. The output is a compression matrix that you can pass to ``compress``.

*Usage*

.. code-block:: bash

    riesling rovir channels.h5 compression-matrix.h5

*Important Options*

* ``--sir=0.05``

    Set the Signal-Interference-Ratio threshold expressed as a fraction of the largest value, below which channels are dropped.

* ``--accept-fov=X,Y,Z``, ``--reject-fov=X,Y,Z``

    The size of the fields-of-view for the signal acceptance (inside) and interference rejection (outside) regions. If ``--reject-fov`` is not given it takes the value of ``--accept-fov``.

downsamp
--------

Remove non-Cartesian samples and trajectory points in order to reconstruct a low resolution down-sampled image.

*Usage*

.. code-block:: bash

    riesling downsamp file.h5 --res=4

*Output*

``file-downsamp.h5`` containing the downsampled non-cartesian data, trajectory and header information.

*Important Options*

* ``--res=R``

    The desired resolution.

* ``--channels=N``

    Compress to N channels.

* ``--energy=E``

    Retain the number of channels required to retain the specified fraction of the variance/energy. Valid values are between 0 and 1.

* ``--pca-read=ST,SZ``

    Take the samples for PCA from `ST` to `ST + SZ` along the read direction.

* ``--pca-traces=ST,SZ,STRIDE``

    Take the samples for PCA from `ST` to `ST + SZ` every `STRIDE` along the spoke direction.


slice
-----

*Usage*

.. code-block:: bash

    riesling slice input.h5 output.h5 --trace=32,32,2

*Important Options*

* ``--channel=start,size,stride``, ``--sample=start,size,stride``, ``--trace=start,size,stride``, ``--slab=start,size,stride``, ``--time=start,size,stride``

    Choose the start element, the number of elements, and the stride between elements for the slice in each dimension

* ``--tps=N``, ``--segment=start,size,stide``

    Additionally split the trace dimension into segments, and then select the slice for the segment dimension. If this option is specified, ``--trace=start,size,stride`` applies within a segment only.
