Frequently Asked Questions
==========================

Understandly, users have questions. Here are some answers.

#. `What is the trajectory scaling?`_
#. `When do I need Sample Density Compensation?`_
#. `Do I need to supply sensitivity maps?`_
#. `What if I have Cartesian data?`_
#. `Help! My data is anisotropic!`_
#. `Which regularizer should I pick?`_
#. `Which algorithm should I pick?`_
#. `Should I denoise or regularize?`_

What is the trajectory scaling?
-------------------------------

The extent of the non-cartesian k-space is scaled as -N/2 to N/2 in each dimension where N is the nominal matrix size. This matches the scaling in BART. Note that the cartesian indices run from -N/2 to N/2-1, which is determined by the definition of the FFT. If you have a truly symmetric non-cartesian trajectory then one k-space sample will be dropped at the positive edge of each dimension. This is correct behaviour for the FFT.

When do I need Sample Density Compensation?
-------------------------------------------

You don't. As of version 1, ``riesling`` uses pre-conditioned iterative algorithms everywhere, including for the basic NUFFT. SDC factors are surprisingly difficult to calculate correctly or quickly, whereas Frank Ong's k-space preconditioner is correct and fast. See `F. Ong, M. Uecker, and M. Lustig, ‘Accelerating Non-Cartesian MRI Reconstruction Convergence Using k-Space Preconditioning’, IEEE Trans. Med. Imaging, vol. 39, no. 5, pp. 1646–1654, May 2020<https://ieeexplore.ieee.org/document/8906069/>`_.

Do I need to supply sensitivity maps?
-------------------------------------

For most data you do not. ``riesling`` will generate the maps by autocalibration from the central region of k-space. This method works extremely well with most data, including cartesian scans. You can generate the maps for viewing using the ``sense-calib`` and ``sense-maps`` commands. The sensitivities calculated from ``sense-calib`` are stored as kernels in k-space to reduce disk space.

If your data contains large holes at the center of k-space (e.g. ZTE data) then you will likely want to obtain sensitivity kernels from another dataset with ``sense-calib``.

What if I have Cartesian data?
------------------------------

``riesling`` handles Cartesian data, but you still need to supply a trajectory. This is likely to be each line of k-space stored as a trace running from -N/2 to (N-1)/2. I then recommend addding ``--osamp=1 --tophat`` which will disable the usual gridding kernel and apodization, such that the NUFFT is effectively a plain FFT.

Help! My data is anisotropic!
-----------------------------

Various places in the ``riesling`` code used to make assumptions that both the matrix- and voxel-sizes were isotropic. These assumptions have been removed and anisotropic data should be reconstruct quite happily. Please contact me if it doesn't work. Note that the ``montage`` tool does not understand anisotropic voxels yet so images from that tool may look weird. If you convert to NIFTI the header information should be correct for display.

Which regularizer should I pick?
--------------------------------

This depends very much on your data. For straightforward single-image reconstruction I recommend ``--tgv``. If your data scaling is correct, then λ on the order of 1e-2 or 1e-3 should produce good results. If instead you are reconstructing multiple images together, either from a subspace or temporal reconstruction, then Locally Low-Rank works well. Set ``--llr-patch=5 --llr-win=1`` for the highest quality possible, but note this will be slow to run. Increasing the window size will yield a big speed increase but can produce a blocky appearance in the final images.

Which algorithm should I pick?
------------------------------

There are to my knowledge only two general purpose algorithms for solving such problems across typical regularizers - ADMM and PDHG. The adaptive version of ADMM in ``riesling`` is essentially parameter-free, but requires expensive solves for the inner problem. This makes it very slow to converge. In contrast PDHG requires a step-size which is determined by the largest eigenvalue of the encoding matrix, which must be found by power iteration before running the reconstruction. This eigenvalue will be constant for a particular trajectory, so can be calculated beforehand.

As of version v1.03 I recommend using PDHG with a precomputed eigenvalue. However, because this requires an extra step and is not robust, ADMM remains the default.

Should I denoise or regularize?
-------------------------------

Many MRI papers use a regularized reconstruction, i.e. they minimize ``|Ax - y| + λ|f(x)|`` where ``A`` is the encoding/system matrix. While this is mathematically elegant the current generation of algorithms for solving this problem directly are slower than those for the unregularized problem.
 
An alternative to this approach is to first solve the unregularized reconstruction problem ``min(|Ax̅ - y|)`` and then subsequently solve the denoising problem ``min(|x - x̅| + λ|f(x)|)``. This allows using the efficient LSMR algorithm, which is parameterless and allows preconditioning, for the first step and then PDHG for the second step. Because there is no transform for the data fidelity term, the maximum eigenvalue is simply 1, so an explicit calculation is not required. This can give acceptable results faster than solving the regularized problem directly.
