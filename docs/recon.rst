Reconstruction
==============

This page details the reconstruction commands in RIESLING:

* `recon`_
* `cg`_
* `lsqr`_
* `admm`_
* `tgv`_

These commands combine the operations in :doc:`op` into a pipeline, and then use a specific optimizer to solve the reconstruction problem.

The final image quality depends a great deal on the choice of optimizer and parameters. What works well for one particular problem may not work for another. However, at the time of writing, the ``lsqr`` method is the favourite of the authors (above ``cg``). ``lsqr`` solves the reconstruction problem :raw-latex:`y=Ex` in a least-squares sense directly, instead of solving the normal equations :raw-latex:`E^{\dagger}y=E^{\dagger}Ex`. This allows the use of correct k-space pre-conditioning instead of sample density compensation, which gives fast convergence without inflating noise.

*Shared Options*

* ``--kernel=NN,KB3,KB5,FI3,FI5``

    Choose the gridding kernel. Valid options are NN (see `C. Oesterle, M. Markl, R. Strecker, F. M. Kraemer, and J. Hennig, ‘Spiral reconstruction by regridding to a large rectilinear matrix: A practical solution for routine systems’, Journal of Magnetic Resonance Imaging, vol. 10, no. 1, pp. 84–92, Jul. 1999 <http://doi.wiley.com/10.1002/%28SICI%291522-2586%28199907%2910%3A1%3C84%3A%3AAID-JMRI12%3E3.0.CO%3B2-D>`), KB3 & KB5 (Kaiser-Bessel, see `P. J. Beatty, D. G. Nishimura, and J. M. Pauly, ‘Rapid gridding reconstruction with a minimal oversampling ratio’, IEEE Transactions on Medical Imaging, vol. 24, no. 6, pp. 799–808, Jun. 2005 <http://ieeexplore.ieee.org/document/1435541/>`), and FI3 & FI5 (see `A. H. Barnett, ‘Aliasing error of the exp ⁡ ( β 1 − z 2 ) kernel in the nonuniform fast Fourier transform’, Applied and Computational Harmonic Analysis, vol. 51, pp. 1–16, Mar. 2021 <https://linkinghub.elsevier.com/retrieve/pii/S1063520320300725>`). The numbers after KB/FI refer to the width of the kernel. The default is FI3, the Flat-Iron kernel is marginally faster than the usual Kaiser-Bessel and gives comparable results.

* ``--osamp=S``

    Grid oversampling factor, default 2. In certain situations, namely non-iterative reconstruction and the entire object contained within the FOV, it is possible to reduce this below 2 (see the Beatty paper linked above). For iterative reconstruction, it is generally best to leave this at 2. When using Töplitz embedding it is required that the grid be at least twice the size of the region of support.

* ``--fast-grid``

    The gridding in RIESLING is multi-threaded. To prevent conflicting writes during the adjoint gridding step, each thread writes to a local grid. This increases memory costs by a factor of two, which can be prohibitive for large 3D reconstructions. This option allows threads to write directly to the final grid, at the risk of conflicting writes (memory races). In the typical case of a large grid and a handful of threads, the chances of a conflicting write actually happening are vanishingly small. The impact of such a conflict is a slight error in the gridded k-space. Use at your own risk.

* ``--sense=file.h5``

    Read SENSE maps from the specified file. The dimensions of the SENSE maps must match the reconstruction grid size.

* ``--sense-vol=V, --sense-res=R, --sense-lambda=L`

    Choose the volume, effective resolution and regularization for generating SENSE maps. See :doc:`util` for more information.

* ``--sdc=none,pipe,pipenn,file.h5``

    Choose the Sample Density Compensation method. Will also be applied to generated SENSE maps.

* ``--sdc-pow=P``

    Apply the SDC power trick from `C. A. Baron, N. Dwork, J. M. Pauly, and D. G. Nishimura, ‘Rapid compressed sensing reconstruction of 3D non-Cartesian MRI’, Magnetic Resonance in Medicine, vol. 79, no. 5, pp. 2685–2692, May 2018, <http://doi.wiley.com/10.1002/mrm.26928>`.

* ``--fov=F``, ``--iter-fov=F``

    Set the output FOV (override the matrix/voxel-size in the header info), and set the FOV used during iterations. As part of the pipeline the images are cropped to a region slightly larger than the output FOV, this helps stabilize the maths. The default value is 256 mm, for body reconstructions a larger value may be required.

* ``--mag``

    Output magnitude value images at the end instead of complex.

* ``--basis=basis.h5``

    RIESLING supports sub-space reconstruction using the specified basis vectors.

*Output*

All reconstruction commands will output a file titled ``file-command.h5`` where ``command`` is the name of the particular command. This will contain the final ``image`` dataset. It will also contain the trajectory and header-information in case you wish to sample the image back to k-space.

recon
-----

The ``recon`` command provides basic non-iterative reconstructions. This is useful when you want to run a quick reconstruction to ensure that the data file is in the correct format, but is unlikely to yield optimal image quality.

*Usage*

.. code-block:: bash

    riesling recon file.h5 --rss

*Important Options*

* ``--rss``

    Apply a root-sum-squares channel combination. Do not generate or use SENSE maps.

* ``--fwd``

    Apply the forward operation, i.e. sample through to non-cartesian k-space. Requires SENSE maps to be supplied.

cg
--

Uses the conjugate-gradients optimizer as described in `K. P. Pruessmann, M. Weiger, P. Börnert, and P. Boesiger, ‘Advances in sensitivity encoding with arbitrary k-space trajectories’, Magn. Reson. Med., vol. 46, no. 4, pp. 638–651, Oct. 2001 <http://doi.wiley.com/10.1002/mrm.1241>`.

*Usage*

.. code-block:: bash

    riesling cg file.h5 --toe --max-its=N

*Important Options*

* ``--toe``

    Use Töplitz embedding as described in `C. A. Baron, N. Dwork, J. M. Pauly, and D. G. Nishimura, ‘Rapid compressed sensing reconstruction of 3D non-Cartesian MRI’, Magnetic Resonance in Medicine, vol. 79, no. 5, pp. 2685–2692, May 2018, <http://doi.wiley.com/10.1002/mrm.26928>`. If this option is used, the reconstruction grid must be at least twice as large as the true region of support of your image. This means that if your acquisition FOV did not completely include the object, you likely need to increase ``--osamp`` beyond 2. This option skips the gridding step during iterations by calculating a transfer function, hence only requiring a Fourier Transform to cartesian k-space.

* ``--thresh=T``, ``--max-its=N``

    Termination conditions. The threshold is applied to the normalized residual. With Density Compensation, reasonable quality images can be obtained in around 8 iterations.

lsqr
----

As described above, ``lsqr`` is an algorithm for solving non-square systems of equations without forming the normal equations. This keeps the condition number low, and allows correct pre-conditioning to be applied in k-space. However, it cannot use Töplitz embedding. This means that individual iterations are slower, but typically fewer of them are needed to reach convergence compared to ``cg``.

*Usage*

.. code-block:: bash

    riesling lsqr file.h5 --pre --atol=1e-4 --sdc=none

*Important Options*

* ``--pre``

    Use Ong's single-channel pre-conditioner (see `F. Ong, M. Uecker, and M. Lustig, ‘Accelerating Non-Cartesian MRI Reconstruction Convergence Using k-Space Preconditioning’, IEEE Trans. Med. Imaging, vol. 39, no. 5, pp. 1646–1654, May 2020, <https://ieeexplore.ieee.org/document/8906069/>`). Highly recommended, likely to become the default.

* ``--sdc=none``

    If using Ong's preconditioner you should switch SDC off. How these arguments are structured is likely to change in a future version.

* ``--atol=A``, ``--btol=B``

    Termination conditions. Determine the absolute and relative residual sizes for termination.

* ``--lambda=L``

    Tikohonov regularization parameter for the reconstruction problem (not for the SENSE maps). Doesn't seem to help much.

admm
----

Uses the Alternating Directions Method-of-Multipliers, also known as an Augmented Lagrangian method, to add a regularizer to the reconstruction problem. Currently the only regularizer available is Locally Low-Rank, which is only useful when reconstructing a multi-frame / basis dataset. By default the inner optimizer is LSQR. See `J. I. Tamir et al., ‘T2 shuffling: Sharp, multicontrast, volumetric fast spin‐echo imaging’, vol. 77, pp. 180–195, 2017 <https://onlinelibrary.wiley.com/doi/abs/10.1002/mrm.26102>`.

*Usage*

.. code-block:: bash

    riesling admm file.h5 --basis=basis.h5 --pre --sdc=none --rho=1.0 --lambda=0.1

*Important Options*

* ``--pre``

    Use pre-conditioning (see ``lsqr`` above).

* ``--cg``

    Use CG instead of LSQR for the inner loop.

* ``--rho=P``

    Coupling factor for ADMM. Values of 1.0 seem to work, and will be adjusted inside the algorithm according to some heuristics if deemed sub-optimal.

* ``--lambda=L``

    Regularization parameter (currently only LLR implemented). See the ``reg`` command in :doc:`util` for further details.

tgv
---

This command uses Total Generalized Variation regularization to improve image quality. See `Knoll, F., Bredies, K., Pock, T. & Stollberger, R. Second order total generalized variation (TGV) for MRI. Magnetic Resonance in Medicine 65, 480–491 (2011).<http://doi.wiley.com/10.1002/mrm.22595>` It uses a different optimization algorithm to ``admm`` and hence is not implemented there. The regularization only applies in the spatial dimensions.

*Usage*

.. code-block:: bash

    riesling tgv file.h5 --alpha=2.e-5

*Important Options*

* ``--alpha=N``

    Regularization parameter. 2e-5 seems to be a magic value and should probably be the default.

* ``--step=S``

    Inverse of the gradient descent step-size taken. Smaller values can lead to faster convergence at the risk of oscillations/artefacts.

* ``--reduce=R``

    Reduce the regularization factor by this factor over the iterations. Can prevent over-smoothing. Default is 0.1.

