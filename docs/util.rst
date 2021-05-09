Utilities
=========

RIESLING includes a number of utilities that either implement functionality that is not strictly reconstruction or assist with basic file management.

Sensitivities
-------------

Estimating channel sensitivities is a key step in modern reconstruction methods. The ``sense`` command allows you to run the direct sensitivity extraction step that is incorporated into the main reconstruction commands and save the results, either for inspection or for use across multiple images. As described in :doc:`recon`, Tikhonov regularization can be applied with ``--lambda``. You must set the ``--fov`` option to match the FOV that the channels will be combined at - for the iterative recon methods this is equal to the ``--iter_fov`` value, which defaults to 256 mm.

``riesling espirit`` implements the popular ESPIRiT method for estimating channel sensitivities that exploits the correlations between channels in k-space. Important options are ``--kRad``, which sets the initial k-space kernel radius and ``--calRad`` which determines the size of the calibration region. RIESLING defines the calibration region as the cubic region with "radius" of twice the kernel radius plus the calibration radius, i.e. the calibration radius is expanded by the kernel width. The default value is one, or one plus the dead-time gap if the data has a gap. ``--thresh`` defines the threshold for retaining kernels after the first SVD step. Any kernel/singular vector with a singular value (as a fraction of the first singular value) above the threshold will be retained.

Pre-calculation
---------------

In addition to pre-calculating sensitivities, it is also possible to pre-calculate sample densities using the ``--sdc`` command. All the reconstruction options (``--kb``, ``--kw``, ``--os``) must match to the settings you will use for the actual reconstruction or you will get artefacts. This command is useful if you will be running many reconstructions with the same trajectory.

The first time you run RIESLING with a new trajectory it will be fairly slow while it "plans" the Fourier transforms. This is a feature of the FFTW library that RIESLING uses internally. In short, FFTW attempts different strategies for any given FFT, measures which is fastest, and then saves this "wisdom" for future use. RIESLING stores these in a file called ``.riesling-wisdom`` in the user's home directory. Similarly to sample densities, the FFT settings can be re-used for any future FFT of the same size. The size of the FFT in RIESLING is controlled by the trajectory and the ``--os`` option. You can run ``riesling plan`` with a particular ``.h5`` file and ``--os`` value to force the planning before you run any real reconstructions.

Compression
-----------

Modern MR scanners are often equipped with multi-channel receive coils with a high number of elements. This increases memory requirements, both on disk and in RAM, and can make steps in the reconstruction ill-conditioned. It is hence advisable to compress the the raw data to a smaller number of virtual channels before running the reconstruction. This step should be carried out first before any subsequent operations, e.g. sensitivity estimation. The ``compress`` command in RIESLING implements basic PCA coil-compression, with the number of output virtual channels specified by the ``--cc`` option. We have not implemented other methods, such as Geometric Coil Compression, because they rely on properties of Cartesian sequences that we cannot rely on for non-Cartesian.

Gridding
--------

``riesling grid`` will carry out only the first step of the NUFFT, i.e. it will grid non-Cartesian k-space to Cartesian (or vice versa) and save the result. This can be useful to check that a dataset has been acquired correctly.

To diagnose trajectory and sample density issues, you can instead use ``riesling traj``. This will apply the gridding and sample density compensation to a set of ones, allowing you to see if the Cartesian k-space has an even weighting.

Simulations
-----------

``riesling phantom`` will produce a simulated image, useful for experimenting with reconstruction settings. Currently spherical and Shepp-Logan phantoms with simple multi-channel coil sensitivities are implemented. It is advisable to use a high grid oversampling rate to minimise rasterization errors. The trajectory can be read from a ``.h5`` file with the ``--traj`` command, otherwise an Archimedean 3D spiral trajectory will be used.

Data Tools
----------

``riesling split`` will split out a single volume from a multi-volume ``.h5`` file, and will separate the low- and high-resolution k-space trajectories if they are present.

Dead-time Gap Filling
---------------------

``riesling zinfandel`` implements an experimental ZTE dead-time gap filling method based on 1D GRAPPA. This will be the subject of a future publication.

References
----------

1. Yeh, E. N. et al. Inherently self-calibrating non-cartesian parallel imaging. Magnetic Resonance in Medicine 54, 1–8 (2005).
2. Uecker, M. et al. ESPIRiT-an eigenvalue approach to autocalibrating parallel MRI: Where SENSE meets GRAPPA. Magnetic Resonance in Medicine 71, 990–1001 (2014).
3. Zwart, N. R., Johnson, K. O. & Pipe, J. G. Efficient sample density estimation by combining gridding and an optimized kernel: Efficient Sample Density Estimation. Magn. Reson. Med. 67, 701–710 (2012).
4. Wong, S. T. S. & Roos, M. S. A strategy for sampling on a sphere applied to 3D selective RF pulse design. Magnetic Resonance in Medicine 32, 778–784 (1994).
