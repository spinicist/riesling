Reconstruction
==============

There are currently three different reconstruction commands provided in RIESLING - ``recon``, ``cg`` and ``tgv``. More may be added in future.

Non-iterative
-------------

The ``recon`` command provides basic non-iterative reconstructions. This is useful when you want to run a quick reconstruction to ensure that the data file is in the correct format, but is unlikely to yield optimal image quality. However it will be useful to describe some of the command options here, because they are shared with ``cg`` and ``tgv``.

By default ``recon`` will output complex-valued images in ``.h5`` format that have been combined using channel sensitivities extracted from the scan. If you are only interested in the magnitude images, add ``--mag`` to the command-line. If you only want a root-sum-squares reconstruction, add ``--rss``. If you would like NIFTI images as output, add ``--oft=nii``.

Non-cartesian MRI data requires a Non-Uniform FFT (NUFFT) instead of a simple FFT for conversion between k-space and image space. The NUFFT consists of a gridding step and then a normal FFT. The gridding step in RIESLING is controlled by the oversampling and kernel options.

``--os=X`` controls the grid oversampling ratio. The default value is 2, but this is very memory intensive (as it is a 3D grid, 2x oversampling requires 8 times the memory of the native matrix size). Reducing the over-sampling to a value of 1.3 leads to a reduced memory foot-print for little impact in image quality. 

``--kb`` selects a Kaiser-Bessel kernel instead of the default nearest-neighbour kernel. Kaiser-Bessel is the default in most other toolboxes and may become the default in RIESLING in the future. The improvement in image quality from KB over NN is less marked in 3D imaging compared to 2D. The ``--kw`` option controls the kernel width, the default value is 3 and is sufficient for most applications.

The final gridding option is ``--fast-grid``. RIESLING uses a parallelized gridding operation. In order to avoid multiple threads writing to the same Cartesian k-space point, the trajectory is sorted by Cartesian k-space location and threads each process their own chunk of the sorted co-ordinates. However, there may still be race conditions at the edge of each chunk, particularly for small images with highly oversampled k-space centers. To prevent this, each thread uses it's own workspace, and then these are combined into the final grid at the end. This process is thread-safe but doubles the memory requirements for the gridding operations. The ``--fast-grid`` option makes the threads write directly into the final grid, reducing memory consumption but at the risk of some writes into the grid being overwritten. When gridding high resolution images on a small number of threads, e.g. 10 or fewer, the probability of a race condition is vanishingly small. Use at your own risk.

The gridding step can also compensate for the increased density of samples in the oversampled central k-space region with most non-Cartesian trajectories. This step is often ommitted in 2D non-Cartesian iterative reconstructions, but is essential for reasonable convergence in 3D non-Cartesian reconstruction. The ``--sdc`` option controls the sample density compensation method - valid values are "none" to turn it off, "radial" for analytic radial densities, and "pipe" for the iterative method of Pipe et al. The default is "pipe". It is also possible to pre-calculate the densities of a given trajectory using the ``riesling sdc`` command and then pass in the resulting file, i.e. ``--sdc=file-sdc.h5``. The weighting of the compensation can be reduced using the ``--sdc_exp`` option - the exponent is applied to all k-space densitites equally and values between 0 and 1 make sense.

Due to the oversampled central region in most non-Cartesian trajectories reasonable channel sensitivities can be extracted directly from the data. This is the default option. Tikhonov regularization can be applied to the sensitivities using the ``--lambda`` option (the value should be approximately the same as the background noise in the ``--rss`` reconstruction). In a multi-volume reconstruction, the sensitivities are taken from the last volume by default but can be specified using ``--senseVolume``, or can be taken from an external file using ``--sense``.

You can apply a basic Tukey filter to the final image k-space using ``--tukey_start``, ``--tukey_end`` and ``--tukey_height``. The start and end are defined as the fractional radius in k-space, i.e. 0.5 and 1. The height option is specified at the end radius and should be between 0 (heavy filtering) and 1 (no filtering). Finally, if you want to expand (or contract) the field-of-view of an image, for instance with a read-oversampled acquisition, then use the ``--fov`` option.

Iterative
---------

The workhorse reconstruction tool in RIESLING is ``cg``, which runs an un-regularized cgSENSE reconstruction. For speed RIESLING uses a Töplitz embedding strategy. This uses the gridding method to calculate the k-space transfer function on the Cartesian grid. After the initial gridding from non-Cartesian to Cartesian grid, each iteration only requires SENSE combination/expansion, the forwards/reverse FFT, and a multiplication of Cartesian k-space by the transfer function.

The additional options added for ``cg`` control the iterations strategy. ``--max_its`` specifies the maximum number of iterations. cgSENSE image quality often benefits from early stopping of the iterations, which is an implicit form of regularization as it prevents the algorithm from over-fitting noise. The default value is 8, with correct density compensation reasonable images can often be obtained in only 4. You can also specify a threshold to terminate the iterations using ``--thresh``. The default value is 1e-10 which is very strict and rarely reached. Values of ``1e-3`` or so would lead to early stopping.

Finally, ``cg`` adds an additional ``--iter_fov`` option which controls the field-of-view cropping used during the iterations. This needs to be larger than the final FOV to avoid aliasing and edge effects. The default value is 256 mm which is sufficient for most brain reconstructions. Note that if you pre-compute sensitivities, their FOV must match this value.

The ``tgv`` command uses Total Generalized Variation regularization to improve image quality. It uses a different optimization algorithm to ``cg`` which is noticeable slower, but still reasonable. It adds three more options. ``--alpha`` controls the initial regularization level. The default is 1e-5, better results can often be obtained with 2e-5. ``--reduce`` will reduce the regularization over the course of the iterations, which can prevent over-smoothing. ``--step`` controls the gradient-descent step size and is specified as an inverse, i.e. a value of 8 results in a step-size of 1/8th the gradient. Smaller values (larger step sizes) give faster convergence but can lead to artefacts.

References
----------

1. Fessler, J. A. & Sutton, B. P. Nonuniform fast fourier transforms using min-max interpolation. IEEE Transactions on Signal Processing 51, 560–574 (2003).
2. Beatty, P. J., Nishimura, D. G. & Pauly, J. M. Rapid gridding reconstruction with a minimal oversampling ratio. IEEE Transactions on Medical Imaging 24, 799–808 (2005).
3. Oesterle, C., Markl, M., Strecker, R., Kraemer, F. M. & Hennig, J. Spiral reconstruction by regridding to a large rectilinear matrix: A practical solution for routine systems. Journal of Magnetic Resonance Imaging 10, 84–92 (1999).
4. Zwart, N. R., Johnson, K. O. & Pipe, J. G. Efficient sample density estimation by combining gridding and an optimized kernel: Efficient Sample Density Estimation. Magn. Reson. Med. 67, 701–710 (2012).
5. Pruessmann, K. P., Weiger, M., Börnert, P. & Boesiger, P. Advances in sensitivity encoding with arbitrary k-space trajectories. Magn. Reson. Med. 46, 638–651 (2001).
6. Knoll, F., Bredies, K., Pock, T. & Stollberger, R. Second order total generalized variation (TGV) for MRI. Magnetic Resonance in Medicine 65, 480–491 (2011).
