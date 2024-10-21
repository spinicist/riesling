Reconstruction
==============

The main reconstruction tool you should use is `riesling recon-rlsq`_. This solves the regularized least-squares reconstruction problem using the Alternating-Directions Method-of-Multipliers algorithm. It supports several common regularizers including L1-wavelets, Total Variation and Total Generalized Variation. To run an unregularized reconstruction use `riesling recon-lsq`_. ``recon-lsq`` and ``recon-rlsq`` will self-calibrate sensitivity maps from the input data if no maps are supplied. These can be calculated explicitly with the `sense-calib`_ command, see there for details of the calibration.

* `recon-lsq`_
* `recon-rlsq`_
* `recon-rss`_
* `sense-calib`_

*Common Options*

* ``--basis=basis.h5``

    RIESLING supports sub-space reconstruction using the specified basis vectors.

* ``--scale=otsu/bart/S``

    This option is ignored for ```rss``` and ``sense``. Specify the scaling of the data during reconstruction. This is important for the regularized reconstructions as it means values of λ will be comparable between different datasets. The aim is to have voxel intensities of about 1 in signal regions. The default method is to perform a SENSE reconstruction and then normalise to the median foreground value determined with Otsu's method. An option to replicate the scaling used in BART is provided. Finally the scaling can be fixed to a known reasonable value. This is a multiplicative scaling.

* ``--kernel=NN,KB2,KB4,KB6,ES2,ES4,ES6``

    Choose the gridding kernel. Valid options are:
    
    ** NN (nearest-neighbour), see `C. Oesterle, M. Markl, R. Strecker, F. M. Kraemer, and J. Hennig, ‘Spiral reconstruction by regridding to a large rectilinear matrix: A practical solution for routine systems’, Journal of Magnetic Resonance Imaging, vol. 10, no. 1, pp. 84–92, Jul. 1999 <http://doi.wiley.com/10.1002/%28SICI%291522-2586%28199907%2910%3A1%3C84%3A%3AAID-JMRI12%3E3.0.CO%3B2-D>`_.
    
    ** KB2/KB4/KB6 Kaiser-Bessel kernel with width 2/4/6. See `P. J. Beatty, D. G. Nishimura, and J. M. Pauly, ‘Rapid gridding reconstruction with a minimal oversampling ratio’, IEEE Transactions on Medical Imaging, vol. 24, no. 6, pp. 799–808, Jun. 2005 <http://ieeexplore.ieee.org/document/1435541/>`_
    
    ** ES2/ES4/ES6 Exponential of a Semi-circle kernel with width 2/4/6. See `A. H. Barnett, ‘Aliasing error of the exp ⁡ ( β 1 − z 2 ) kernel in the nonuniform fast Fourier transform’, Applied and Computational Harmonic Analysis, vol. 51, pp. 1–16, Mar. 2021 <https://linkinghub.elsevier.com/retrieve/pii/S1063520320300725>`_
    
    The default is ES4 which is marginally faster than the usual Kaiser-Bessel and gives comparable results. Wider kernels provide a marginal increase in image quality at the expense of much slower runtimes. ES2 usually gives acceptable image quality and can be much faster.

* ``--osamp=S``

    Grid oversampling factor, default 1.3. See `P. J. Beatty, D. G. Nishimura, and J. M. Pauly, ‘Rapid gridding reconstruction with a minimal oversampling ratio’, IEEE Transactions on Medical Imaging, vol. 24, no. 6, pp. 799–808, Jun. 2005 <http://ieeexplore.ieee.org/document/1435541/>`_.

* ``--fov=F``

    Set the reconstruction FOV. A new matrix size will be calculated using the header voxel-size information.In situations where there is significant signal outside the nominal FOV, but the data was acquired oversampled, then this can be used to prevent aliasing artefacts and improve image quality. `C. A. Baron, N. Dwork, J. M. Pauly, and D. G. Nishimura, ‘Rapid compressed sensing reconstruction of 3D non-Cartesian MRI’, Magnetic Resonance in Medicine, vol. 79, no. 5, pp. 2685–2692, May 2018 <http://doi.wiley.com/10.1002/mrm.26928>`_.

* ``--lowmem``

    3D non-cartesian reconstructions can consume large amounts of memory. By default RIESLING will reconstruct all channels simultaneously, requiring that both the oversampled grid and the sensitivity maps for each are held in RAM. Enabling this option swaps to a scheme where only one grid and sensitivity map are kept in RAM. This requires repeating the NUFFT calculations for each channel, trading memory size for reconstruction speed.

* ``--precon=none/kspace/file``

    Choose a diagonal k-space preconditioner. The default is Frank Ong's preconditioner. See `F. Ong, M. Uecker, and M. Lustig, ‘Accelerating Non-Cartesian MRI Reconstruction Convergence Using k-Space Preconditioning’, IEEE Trans. Med. Imaging, vol. 39, no. 5, pp. 1646–1654, May 2020<https://ieeexplore.ieee.org/document/8906069/>`_.

* ``--pre-bias=N``

    In a sub-space reconstruction it is possible for the preconditioner calculation to contain divide-by-zero problems. This option adds a bias to the calculation to prevent this causing problems. The default value is 1.

recon-lsq
---------

Reconstructs the image as the least-squares solution to an inverse problem. The system matrix includes sensitivity maps and the NUFFT. The LSMR algorithm is used as this does not require forming the normal equations, keeping the condition number low. In combination with pre-conditioning in k-space this ensures convergence in a handful of iterations.

*Usage*

.. code-block:: bash

    riesling recon-lsq input.h5 output.h5

*Important Options*

* ``--max-its=N``, ``--atol=A``, ``--btol=B``, ``--ctol=C``

    Termination conditions. Reasoable image quality can be obtained in less than five iterations. The a and b tolerances are relative to how accurate the solution has become, c is a tolerance on the condition number of the system.

* ``--lambda=L``

    Apply basic Tikohonov/L2 regularization to the reconstruction.

recon-rlsq
----------

Uses the Alternating Directions Method-of-Multipliers to add regularizers to the least-squares reconstruction problem. This is similar to the BART ``pics`` command. See `S. Boyd, ‘Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers’ doi: 10.1561/2200000016 <http://www.nowpublishers.com/article/Details/MAL-016>`_

*Usage*

.. code-block:: bash

    riesling recon-rlsq input.h5 output.h5 --tgv=1e-3

*Important Options*

* ``--max-its=N``, ``--max-its0=N``--atol=A``, ``--btol=B``, ``--ctol=C``

    These are the same as for ``recon-lsq`` and control the inner loop of the optimization (the x update step). As this step is warm-started, the default for `max-its` is 1. However, this may be insufficient to reach a good approximation of the answer on the first outer iteration,so there is an extra `max-its0` option with a default of 4.

* ``--max-outer-its=N``

    The maximum number of ADMM iterations. The default is 20 but a higher number (50 or more) may be required for optimal image quality.

* ``--eps=E``

    Primal and dual convergence tolerance for ADMM. Default value is 0.01.

* ``--rho=P``

    Coupling factor for ADMM. The default value of 1 is robust, and will be adjusted inside the algorithm according to `ADMM Penalty Parameter Selection by Residual Balancing <http://arxiv.org/abs/1704.06209>`_.

* ``--scale=bart/otsu/S``

    The optimal regularization strength λ depends both on the particular regularizer and the typical intensity values in the unregularized image. To make values of λ roughly comparable, it is usual to scale the data such that the intensity values are approximately 1 during the optimization (and then unscale the final image). By default ``riesling`` will perform a NUFFT and then use Otsu's method to find the median foreground intensity as the scaling factor (specify ``otsu`` to make this explicit). The BART automatic scaling can be chosen with ``bart``. Alternately a fixed numeric *multiplicative* scaling factor can be specified, which will skip the initial NUFFT. If you already know the approximate scaling of your data (from a test recon), this option will be the fastest.

*Regularization Options*

Multiple regularizers can be specified simultaneously with ADMM, each with a different regularization strength λ and options. At least one regularizer must be specified, there is no default option at present.

* ``--l1=λ``

    Basic L1 regularization in the image domain, i.e. λ|x|.

* ``--nmrent=λ``

    Similar to L1 regularization. See `Daniell and Hore <https://linkinghub.elsevier.com/retrieve/pii/0022236489901170>`_. `Not recommended <https://onlinelibrary.wiley.com/doi/10.1002/mrm.1910140103>`_.

* ``--tv=λ``

    Classic `Total Variation <https://linkinghub.elsevier.com/retrieve/pii/016727899290242F>`_ regularization, i.e. λ|∇x|

* ``--tgv=λ``, ``--tgvl2=λ``

    `Total Generalized Variation <http://doi.wiley.com/10.1002/mrm.22595>`_ and `TGV on the L2 voxelwise norm <http://ieeexplore.ieee.org/document/7466848/>`_. The latter is useful for multichannel images. Note that due to the way the TGV problem is formulated, it consumes significantly more memory and is slower than TV for the same data.

* ``--llr=λ``, ``--llr-patch=N``, ``--llr-win=N``, ``--llr-shift``

    `Locally Low-Rank <https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.26102>`_ regularization. The patch size determines the region to calculate the SVD over, the window size determines the region that is copied to the output image. Set the window size to 1 to calculate an SVD for each output voxel. Set the window size equal to the patch size to use the entire patch. The ``--llr-shift`` option employs the random patch shifting strategy, this may not converge.

* ``--wavelets=λ``, ``--wavelet-width=W``, ``--wavelet-dims=0,1,1,1``

    L1-wavelets of width W (default 6). The number of levels is the maximum possible. Which of the basis,X,Y,Z dimensions to be transformed can be specified with the ``--wavelet-dims`` option.

recon-rss
---------

Perform a basic reconstruction using root-sum-of-squares channel combination. Very fast but worst image quality. Does not calculate or use sensitivity maps. Useful for testing.

*Usage*

.. code-block:: bash

    riesling recon-rss input.h5 output.h5

sense-calib
-----------

Sensitivity maps are an integral part of any reconstruction from a multi-channel coil. Calculating high quality sensitivity maps is a difficult and open research question for two reasons. First, the multi-channel reconstruction problem is ill-posed as there is no unique solution (if the sensitivities are multiplied and the image divided by an arbitrary field the same data will result), and second because sensitivities exist in the background of the image where we cannot acquire signal.

RIESLING estimates sensitivities assuming that a fully-sampled calibration region with consistent contrast has been acquired in the data. This is true for the majority of non-cartesian sequences, see `E. N. Yeh et al., ‘Inherently self-calibrating non-cartesian parallel imaging’, Magnetic Resonance in Medicine, vol. 54, no. 1, pp. 1–8, Jul. 2005, <http://doi.wiley.com/10.1002/mrm.20517>`_, and this step is hence incorporated into the reconstruction commands. However, there are many situations where it is beneficial to calculate the sensitivities up-front, potentially from alternate data. There is hence an explicit ``sense-calib`` command for this. All the relevant options to this command are also exposed for the reconstruction commands.

Note that RIESLING calculates and stores the sensitivity kernels in k-space, not the maps themselves. If you want to see the maps, a separate ``sense-maps`` command is provided to convert between them.

The FOV and oversampling used in the calibration must match your reconstruction.

The algorithm used by RIESLING consists of these steps:

1. Reconstruct low-resolution images for each channel from a fully-sampled calibration region (inverse NUFFT).
2. Obtain a reference image either from fully-sampled single channel data or by taking the root-sum-squares across the multi-channel images.
3. Solve the inverse problem :math:`c = RFPs` where :math:`s` are the sensitivity kernels in k-space, :math:`c` are the channel images, :math:`P` is a padding operator, :math:`F` is an FT, and :math:`R` is an operator that multiplies each channel by the reference image.

To ensure the maps are smooth and have support in the background region, the forward model is modified to incorporate regularization with a Sobolev Norm term :math:`λW = (1 + |k|^2)^{l/2}` (where :math:`k` is k-space co-ordinate, i.e. penalises high frequency terms) and a mask :math:`M` over the object:

.. math::
    c' = A s\\
    c' = \begin{bmatrix}
        c\\
        0
    \end{bmatrix}\\
    A = \begin{bmatrix}
        M R F P\\
        λW
    \end{bmatrix}

This problem is badly conditioned, and even with a preconditioner can take approximately 100 iterations to converge. However due to the small matrix sizes this should only take a few seconds. See `H. C. M. Holme, S. Rosenzweig, F. Ong, R. N. Wilke, M. Lustig, and M. Uecker, ‘ENLIVE: An Efficient Nonlinear Method for Calibrationless and Robust Parallel Imaging’, Scientific Reports, vol. 9, no. 1, Dec. 2019, <http://www.nature.com/articles/s41598-019-39888-7>`_ for the regularizer.

*Usage*

.. code-block:: bash

    riesling sense-calib input.h5 kernels.h5

*Important Options*

* ``--ref=reference.h5``

    Use the supplied data to reconstruct the reference image (i.e. from a body coil acquisition) instead of using the root-sum-squares of the channels.

* ``--sense-lambda=λ``

    The amount of regularization to apply to the sensitivities. Over regularization will result in the per-voxel sensitivities reducing.

* ``--sense-l=L``

    The L parameter to the Sobolev Norm weights. Higher numbers increase the regularization strength in a highly non-linear fashion.

* ``--sense-res=R``

    The resolution of the initial reconstructions for the sensitivity maps. Because sensitivities are generally agreed to be smooth, only a low resolution reconstruction is required and the default is 6mm isotropic. However, the resulting images must have a sufficiently large matrix size to extract the kernels from.

* ``--sense-width=K``

    The width of the sensitivity kernels in k-space on the nominal grid. The value specified here will be mulitipled by the oversampling factor to produce the final kernel size. Hence, if you override the default oversampling in the main reconstruction you must also do so here.

* ``--sense-tp=T``

    If the input data contains multiple timepoints, use this one to calculate the sensitivities (default is first volume).