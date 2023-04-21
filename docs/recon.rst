Reconstruction
==============

The main reconstruction tool you should use is ``riesiling admm``. This solves the regularized least-squares reconstruction problem using the Alternating-Directions Method-of-Multipliers algorithm. It supports several common regularizers including L1-wavelets, Total Variation and Total Generalized Variation.

If you wish to run an unregularized reconstruction, both the classic conjugate-gradients and the more modern LSMR algorithms are available. LSMR is recommended for 3D non-cartesian data as it supports pre-conditioning.

Finally, basic non-iterative reconstructions are available with the `rss` and `sense` commands.

* `rss`_
* `sense`_
* `cg`_
* `lsqr / lsmr`_
* `admm`_

*Common Options*

* ``--basis=basis.h5``

    RIESLING supports sub-space reconstruction using the specified basis vectors.

* ``--scale=otsu/bart/S``

    This option is ignored for ```rss``` and ``sense``. Specify the scaling of the data during reconstruction. This is important for the regularized reconstructions as it means values of λ will be comparable between different datasets. The aim is to have voxel intensities of about 1 in signal regions. The default method is to perform a SENSE reconstruction and then normalise to the median foreground value determined with Otsu's method. An option to replicate the scaling used in BART is provided. Finally the scaling can be fixed to a known reasonable value. This is a multiplicative scaling.

* ``--kernel=NN,KB3,KB5,KB7,ES3,ES5,ES7``

    Choose the gridding kernel. Valid options are:
    
    ** NN (nearest-neighbour), see `C. Oesterle, M. Markl, R. Strecker, F. M. Kraemer, and J. Hennig, ‘Spiral reconstruction by regridding to a large rectilinear matrix: A practical solution for routine systems’, Journal of Magnetic Resonance Imaging, vol. 10, no. 1, pp. 84–92, Jul. 1999 <http://doi.wiley.com/10.1002/%28SICI%291522-2586%28199907%2910%3A1%3C84%3A%3AAID-JMRI12%3E3.0.CO%3B2-D>`_.
    
    ** KB3/KB5/KB7 Kaiser-Bessel kernel with width 3/5/7. See `P. J. Beatty, D. G. Nishimura, and J. M. Pauly, ‘Rapid gridding reconstruction with a minimal oversampling ratio’, IEEE Transactions on Medical Imaging, vol. 24, no. 6, pp. 799–808, Jun. 2005 <http://ieeexplore.ieee.org/document/1435541/>`_
    
    ** ES3/ES5/ES7 Exponential of a Semi-circle kernel with width 3/5/7. See `A. H. Barnett, ‘Aliasing error of the exp ⁡ ( β 1 − z 2 ) kernel in the nonuniform fast Fourier transform’, Applied and Computational Harmonic Analysis, vol. 51, pp. 1–16, Mar. 2021 <https://linkinghub.elsevier.com/retrieve/pii/S1063520320300725>`_
    
    The default is ES3 which is marginally faster than the usual Kaiser-Bessel and gives comparable results. Wider kernels provide a marginal increase in image quality at the expense of much slower runtimes.

* ``--osamp=S``

    Grid oversampling factor, default 2. In certain situations, namely non-iterative reconstruction and the entire object contained within the FOV, it is possible to reduce this below 2 (see the Beatty paper linked above). For iterative reconstruction, it is generally best to leave this at 2. When using Töplitz embedding it is required that the grid be at least twice the size of the region of support.

* ``--bucket-size=B``

    Gridding is divided into buckets to enable parallelization. This controls the bucket size. You may be able to obtain better core utilization by tweaking it slightly. See `A. H. Barnett, J. F. Magland, and L. af Klinteberg, ‘A parallel non-uniform fast Fourier transform library based on an “exponential of semicircle” kernel’. arXiv, Apr. 08, 2019. <http://arxiv.org/abs/1808.06736>`_

* ``--sense=file.h5``

    Read SENSE maps from the specified file. The dimensions of the SENSE maps must match the reconstruction grid size.

* ``--sense-vol=V, --sense-frame=F, --sense-res=R, --sense-lambda=L, --sense-fov=F`

    If SENSE maps are not specified, the maps will be generated directly from the input data. See `[1] E. N. Yeh et al., ‘Inherently self-calibrating non-cartesian parallel imaging’, Magnetic Resonance in Medicine, vol. 54, no. 1, pp. 1–8, Jul. 2005, <http://doi.wiley.com/10.1002/mrm.20517>`_.
    Choose the volume, frame, effective resolution, regularization and field-of-view for generating SENSE maps. The field-of-view should be larger than the final field-of-view for numerical stability. See :doc:`util` for more information.

* ``--fov=F``

    Set the output FOV (override the matrix/voxel-size in the header info).

* ``--sdc=none,pipe,pipenn,file.h5, --sdc-its=N``

    Choose the Sample Density Compensation method. For ``rss``, ``sense`` and ``cg`` this is the default. For ``lsmr``, ``lsqr`` and ``admm`` pre-conditioning is preferred and by default SDC will not be used.

* ``--sdc-pow=P``

    Apply the SDC power trick from `C. A. Baron, N. Dwork, J. M. Pauly, and D. G. Nishimura, ‘Rapid compressed sensing reconstruction of 3D non-Cartesian MRI’, Magnetic Resonance in Medicine, vol. 79, no. 5, pp. 2685–2692, May 2018, <http://doi.wiley.com/10.1002/mrm.26928>`_.

* ``--pre=none/kspace/file``

    Choose a diagonal k-space preconditioner. Only available for ``lsmr``, ``lsqr`` and ``admm``. The default is Frank Ong's preconditioner. See `F. Ong, M. Uecker, and M. Lustig, ‘Accelerating Non-Cartesian MRI Reconstruction Convergence Using k-Space Preconditioning’, IEEE Trans. Med. Imaging, vol. 39, no. 5, pp. 1646–1654, May 2020<https://ieeexplore.ieee.org/document/8906069/>`_.

* ``--pre-bias=N``

    In a sub-space reconstruction it is possible for the preconditioner calculation to contain divide-by-zero problems. This option adds a bias to the calculation to prevent this causing problems. The default value is 1.

*Output*

All reconstruction commands will output a file titled ``file-command.h5`` where ``command`` is the name of the particular command. This will contain the final ``image`` dataset. If you specify ``--keep`` it will also contain the trajectory and header-information in case you wish to sample the image back to k-space.

rss
---

Perform a basic reconstruction using root-sum-of-squares channel combination. Very fast but worst image quality.

*Usage*

.. code-block:: bash

    riesling rss file.h5

sense
-----

Perform a basic reconstruction using SENSE channel combination.

*Usage*

.. code-block:: bash

    riesling sense file.h5

*Important Options*

* ``--fwd``

    Apply the forward operation, i.e. sample through to non-cartesian k-space. Useful for sampling phantoms.

cg
--

Uses the conjugate-gradients optimizer as described in `K. P. Pruessmann, M. Weiger, P. Börnert, and P. Boesiger, ‘Advances in sensitivity encoding with arbitrary k-space trajectories’, Magn. Reson. Med., vol. 46, no. 4, pp. 638–651, Oct. 2001 <http://doi.wiley.com/10.1002/mrm.1241>`_.

*Usage*

.. code-block:: bash

    riesling cg file.h5 --toe --max-its=N

*Important Options*

* ``--thresh=T``, ``--max-its=N``

    Termination conditions. The threshold is applied to the normalized residual. With Density Compensation, reasonable quality images can be obtained in around 8 iterations.

lsqr / lsmr
-----------

These are algorithms for solving non-square systems of equations without forming the normal equations. This keeps the condition number low, and allows correct pre-conditioning to be applied in k-space. The more modern LSMR algorithm is preferred as it takes step to reduce the residual in image space.

*Usage*

.. code-block:: bash

    riesling lsqr file.h5 --atol=1e-4 --sdc=none

*Important Options*

* ``--max-iters=N``, ``--atol=A``, ``--btol=B``, ``--ctol=C``

    Termination conditions. Reasoable image quality can be obtained in as few as four iterations, but high-resolution features from undersampled data typically take a few tens of iterations. The a and b tolerances are relative to how accurate the solution has become, c is a tolerance on the condition number of the system.

* ``--lambda=L``

    Apply basic Tikohonov/L2 regularization to the reconstruction.

admm
----

Uses the Alternating Directions Method-of-Multipliers to add regularizers to the reconstruction problem. This is similar to the BART ``pics`` command. See `S. Boyd, ‘Distributed Optimization and Statistical Learning via the Alternating Direction Method of Multipliers’, FNT in Machine Learning, vol. 3, no. 1, pp. 1–122, 2010, doi: 10.1561/2200000016<http://www.nowpublishers.com/article/Details/MAL-016>`_

*Usage*

.. code-block:: bash

    riesling admm file.h5 --lambda=1e-3 --tgv

*Important Options*

* ``--rho=P``

    Coupling factor for ADMM. The default value of 1 is robust, and will be adjusted inside the algorithm according to some heuristics if deemed sub-optimal.

* ``--lambda=L``

    Regularization strength.

* ``--abstol=A``, ``--reltol=R``

    Set the absolute and relative tolerances for ADMM convergence. The absolute tolerance is generally more important, and will depend on the scaling of your data during reconstruction.

* ``--tgv``

    Apply Total Generalized Varation regularization. See `F. Knoll, K. Bredies, T. Pock, and R. Stollberger, ‘Second order total generalized variation (TGV) for MRI’, Magnetic Resonance in Medicine, vol. 65, no. 2, pp. 480–491, Feb. 2011 <http://doi.wiley.com/10.1002/mrm.22595>`_

* ``--tv``

    Apply Total-Variation regularization. See `L. I. Rudin, S. Osher, and E. Fatemi, ‘Nonlinear total variation based noise removal algorithms’, Physica D: Nonlinear Phenomena, vol. 60, no. 1–4, pp. 259–268, Nov. 1992, doi: 10.1016/0167-2789(92)90242-F.<https://linkinghub.elsevier.com/retrieve/pii/016727899290242F>`_

* ``--llr-patch=N``, ``--llr-win=N``

    Sliding-window Locally Low-Rank regularization. Set the patch-size to enable. See `J. I. Tamir et al., ‘T2 shuffling: Sharp, multicontrast, volumetric fast spin‐echo imaging’, vol. 77, pp. 180–195, 2017 <https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.26102>`_.

* ``--wavelets=N``, ``--width=W``

    L1-wavelets with N levels and width W.

* ``--l1`` / ``--nmrent``

    Basic L1 or NMR entropy (similar to L1) regularization. See `[1] B. Burns, N. E. Wilson, J. K. Furuyama, and M. A. Thomas, ‘Non-uniformly under-sampled multi-dimensional spectroscopic imaging in vivo : maximum entropy versus compressed sensing reconstruction’, NMR Biomed., vol. 27, no. 2, pp. 191–201, Feb. 2014, doi: 10.1002/nbm.3052.<https://onlinelibrary.wiley.com/doi/10.1002/nbm.3052>`_
