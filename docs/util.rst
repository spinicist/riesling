Utilities
=========

RIESLING includes a number of utility commands. These can be broken down into categories.

*Basics*

* `hdr`_
* `meta`_
* `nii`_
* `phantom`_
* `plan`_
* `traj`_

*SENSE maps*

* `sense-calib`_
* `espirit-calib`_

*Density Compensation*

* `sdc`_

*Data manipulation*

* `compress`_
* `downsamp`_
* `split`_

hdr
---

Prints out the header information from a RIESLING ``.h5`` file.

*Usage*

.. code-block:: bash

    riesling hdr file.h5

*Output*

The entire header and meta-data will be written to stdout.

*Important Options*

None!

meta
----

Extracts information from the meta-data or header fields and returns it. Helpful for writing shell scripts that need this information for reconstruction steps.

*Usage*

.. code-block:: bash

    riesling meta file.h5 KEY1 KEY2 ...

*Output*

The desired meta-data will be written to stdout, one field per line.

*Important Options*

None!

The valid header info fields are ``matrix``, ``channels``, ``samples``, ``traces``, ``volumes`` & ``frames``.

nii
---

*Usage*

.. code-block:: bash
    
    riesling nii file.h5 output_file

*Output*

A NIfTI image containing the desired images as separate volumes. The ``.nii`` extension will be added automatically. Currently converting a ``channels`` dataset is not supported.

*Important Options*

* ``--mag``

    Output magnitude images instead of complex.

* ``--dset=NAME``

    Specify the dataset you want to convert. The default is ``image``.

* ``--frame=F``

    Choose a specific frame to convert and discard the others.

* ``--volume=V``

    Choose a specific volume to convert and discard the others.

phantom
-------

Create phantom data, both in image and k-space. Currently RIESLING uses a NUFFT to generate the k-space data and does not generate the phantom data directly in k-space.

*Usage*

.. code-block:: bash
    
    riesling phantom phantom.h5 --matrix=64

*Output*

The specified file (above ``phantom.h5``) containing the phantom image, trajectory and k-space data.

*Important Options*

* ``--matrix=M``

    The matrix size (isotropic matrix assumed).

* ``--shepp_logan``

    Create a Shepp-Logan phantom instead of a boring sphere.

* ``--channels=C``

    Set the number of coil channels.

* ``--rings=N``

    Divide the channels into N rings.

* ``--sense=file.h5``

    Read SENSE maps from a file instead of creating bird-cage sensitivities.

plan
----

RIESLING uses the FFTW library to perform Fourier Transforms. To improve speed RIESLING uses the Wisdom functionality in FFTW. This means the first time you perform an FFT of a particular size, the FFTW library will test different ways of performing the transform and choose the fastest. Hence the first time you run RIESLING on a particular trajectory it will pause while this measurement takes place. Subsequent calls to RIESLING will not pause. You can plan this wisdom up-front for a particular trajectory using this command.

*Usage*

.. code-block:: bash
    
    riesling plan file.h5

*Output*

None.

*Important Options*

* ``--oversamp=S``

    This must match the oversampling that will be used for the reconstruction. Otherwise RIESLING will calculate a different grid size and the results of ``plan`` will be useless.

* ``--time=L``

    Specify a time-limit for planning. May lead to suboptimal results.

traj
----

Performs the gridding step using the trajectory information only, i.e. grids a set of ones instead of the actual k-space data. Useful for producing plots of the trajectory including gridding kernel effects etc.

*Usage*

.. code-block:: bash
    
    riesling traj file.h5 --psf

*Output*

``file-traj.h5`` which will contain the gridded trajectory and optionally the Point Spread Function

*Important Options*

* ``--oversamp=S``

    Grid oversampling factor, default 2.0.

* ``--kernel=K``

    Gridding kernel. Valid options are ``NN`` (Nearest-Neighbour), ``KB3`` & ``KB5`` (Kaiser-Bessel width 3 or 5) and ``ES3`` & ``ES5`` (Flat-Iron width 3 or 5)

* ``--psf``

    Write out the Point Spread Function as well as the trajectory.

sense-calib
-----------

This command is an implmentation of `Yeh, E. N. et al. Inherently self-calibrating non-cartesian parallel imaging. Magnetic Resonance in Medicine 54, 1–8 (2005). <http://doi.wiley.com/10.1002/mrm.20517>`_. Non-cartesian trajectories generally oversampling the center of k-space and hence inherently contain the information necessary to extract SENSE maps. This step is performed by default in the RIESLING pipelines, but if you wish to examine the sensitivities, or use a second file to create them, then you can use this command to write them out explicitly.

*Usage*

.. code-block:: bash

    riesling sense-calib file.h5

*Output*

``file-sense.h5`` containing a dataset ``sense``.

*Important Options*

* ``--sense-vol=N``

    Use the specified volume for SENSE map estimation (default last).

* ``--sense-res=R``

    Calculate the maps at approximately this resolution (default 12 mm).

* ``--sense-lambda=L``

    Apply Tikohonov regularization. See `Ying, L. & Xu, D. On Tikhonov regularization for image reconstruction in parallel MRI. 4. <https://ieeexplore.ieee.org/document/1403345>`_.

* ``--fov=F``

    Calculate the SENSE maps for this FOV. If you are using the maps for subsequent reconstruction, this must match the cropping FOV used during iterations. The default value is 256 mm, which matches the default iterations cropping FOV.

espirit-calib
-------------

An implementation of `Uecker, M. et al. ESPIRiT-an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA. Magnetic Resonance in Medicine 71, 990–1001 (2014).<http://doi.wiley.com/10.1002/mrm.24751>`_ for estimating SENSE maps.

*Usage*

.. code-block:: bash

    riesling espirit-calib file.h5

*Output*

``file-espirit.h5`` containing a dataset ``sense``.

*Important Options*

* ``--sense-vol=N``

    Use the specified volume for SENSE map estimation (default last).

* ``--sense-res=R``

    Calculate the maps at approximately this resolution (default 12 mm).

* ``--fov=F``

    Calculate the SENSE maps for this FOV. If you are using the maps for subsequent reconstruction, this must match the cropping FOV used during iterations. The default value is 256 mm, which matches the default iterations cropping FOV.

* ``--read-start=R``

    Index to start taking samples on traces (to avoid dead-time gap)

* ``--krad=R``

    ESPIRiT kernel radius

* ``--thresh=T``

    Variance threshold to retain kernels (default 0.015)

sdc
---

Non-cartesian trajectories by definition have a non-uniform sample density in cartesian k-space - i.e. there are more samples points per unit volume in some parts of k-space than others. Because the gridding step is a simple convolution, this results in artefactually higher k-space intensities on the cartesian grid. For a non-iterative reconstruction this must be compensated by dividing samples in non-cartesian k-space by their sample density before gridding.

In iterative reconstructions the situation is more complicated, because the forward gridding step does not require density compensation (because after the forward FFT samples are evenly spaced on the cartesian grid). The uneven density affects the condition number of the problem, and hence can lead to slow convergence, but does not fundamentally alter the solution. Hence density compensation is often ommitted in 2D non-cartesian reconstructions. However, in 3D convergence becomes problematic. Strictly speaking, density compensation cannot be inserted into the conjugate-gradients method that is implemented in ``riesling cg``. In practice it can be, and is implemented in RIESLING, but this leads to noise inflation (see `K. P. Pruessmann, M. Weiger, P. Börnert, and P. Boesiger, ‘Advances in sensitivity encoding with arbitrary k-space trajectories’, Magn. Reson. Med., vol. 46, no. 4, pp. 638–651, Oct. 2001 <http://doi.wiley.com/10.1002/mrm.1241>`_). The correct method, implemented in ``riesling lsqr``, is pre-conditioning in k-space.

The sample density is a property of the trajectory and not of the acquired k-space data. Calculating the Sample Density Compensation (SDC) can be time consuming for large trajectories, hence this command exists to pre-calculate it once for a particular trajectory and save it for future use. There are three different methods implemented which are detailed below.

*Usage*

.. code-block:: bash

    riesling sdc sdc.h5 --sdc=pipe

*Output*

``sdc.h5`` containing the density compensation dataset.

*Important Options*

* `--sdc=pipe,pipenn,radial`

    Choose the method to calculate the density compensation. ``pipe`` chooses the method of Pipe, Zwart & Menon. See `N. R. Zwart, K. O. Johnson, and J. G. Pipe, ‘Efficient sample density estimation by combining gridding and an optimized kernel: Efficient Sample Density Estimation’, Magn. Reson. Med., vol. 67, no. 3, pp. 701–710, Mar. 2012 <http://doi.wiley.com/10.1002/mrm.23041>`_. This generates the best results but is slow to compute (requiring 40 iterations on a highly oversampled grid). Hence the default ``pipenn`` method  uses nearest-neighbour gridding to speed up the process, but with a loss of accuracy. Use ``pipe`` for high-quality reconstructions. ``radial`` implements analytic density compensation for 2D and 3D radial trajectories, but from experience this does deal perfectly with undersampling in the spoke direction and the results from ``pipe`` are superior.

* `--os=N`

    Oversampling factor when using ``pipenn``. Should match the oversampling for the final reconstruction.

compress
--------

Reduce the channel count using a coil-compression method.

*Usage*

.. code-block:: bash

    riesling compress file.h5 --pca --channels=12

*Output*

``file-compressed.h5`` containing the compressed non-cartesian data, trajectory and header information.

*Important Options*

* ``--pca``

    Use PCA coil compression. See `Huang, F., Vijayakumar, S., Li, Y., Hertel, S. & Duensing, G. R. A software channel compression technique for faster reconstruction with many channels. Magnetic Resonance Imaging 26, 133–141 (2008). <https://linkinghub.elsevier.com/retrieve/pii/S0730725X07002731>`_.

* ``--channels=N``

    Compress to N channels.

* ``--energy=E``

    Retain the number of channels required to retain the specified fraction of the variance/energy. Valid values are between 0 and 1.

* ``--pca-read=ST,SZ``

    Take the samples for PCA from `ST` to `ST + SZ` along the read direction.

* ``--pca-traces=ST,SZ,STRIDE``

    Take the samples for PCA from `ST` to `ST + SZ` every `STRIDE` along the spoke direction.

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


split
-----

*Usage*

.. code-block:: bash

    riesling split file.h5 --lores=N --sps=S

*Output*

Depends on arguments, but may result in ``file-lores.h5``, ``file-hires.h5`` or files of the form ``file-hires-000.h5``.

*Important Options*

* ``--lores=N``

    Split out the first N traces assuming that they are a low-resolution k-space.

* ``--stride=S``

    Keep only one out of every S traces for further processing (applied after ``--lores``)

* ``--size=N``

    Keep only the first N traces for further processing (applied after ``--lores`` and ``--stride``)

* ``--sps=N``

    Split the hi-res k-space data into multiple files, each containing N traces. If N does not divide the number of traces in the file exactly, the last file will contain the remainder.

* ``--frames=F``, ``--spf=N``

    Add a ``frames`` object to the output header with F frames, each containing N traces. These will be repeated to match the number of traces in the file.
