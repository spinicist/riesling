Frequently Asked Questions
==========================

Understandly, users have questions. Here are some answers.

#. `What is the trajectory scaling?`_
#. `When do I need Sample Density Compensation?`_
#. `Do I need to supply sensitivity maps?`_
#. `What if I have Cartesian data?`_
#. `Help! My data is anisotropic!`_
#. `Which regularizer should I pick?`_
#. `Should I denoise or regularize?`

What is the trajectory scaling?
-------------------------------

The extent of the non-cartesian k-space is scaled as -N/2 to N/2 in each dimension where N is the nominal matrix size. This matches the scaling in BART. Note that the cartesian indices run from -N/2 to N/2-1, which is determined by the definition of the FFT. If you have a truly symmetric non-cartesian trajectory then one k-space sample will be dropped at the positive edge of each dimension. This is correct behaviour for the FFT.

When do I need Sample Density Compensation?
-------------------------------------------

You don't. As of version 1, ``riesling`` uses pre-conditioned iterative algorithms everywhere, including for the basic NUFFT. SDC factors are surprisingly difficult to calculate correctly or quickly, whereas Frank Ong's k-space preconditioner is correct and fast. See `F. Ong, M. Uecker, and M. Lustig, ‘Accelerating Non-Cartesian MRI Reconstruction Convergence Using k-Space Preconditioning’, IEEE Trans. Med. Imaging, vol. 39, no. 5, pp. 1646–1654, May 2020<https://ieeexplore.ieee.org/document/8906069/>`_.

Do I need to supply sensitivity maps?
-------------------------------------

For most data you do not. ``riesling`` will generate the maps by heavily filtering the central region of k-space, performing a NUFFT and then normalizing the resulting channel images by the root-sum-of squares. See `Yeh, E. N. et al. Inherently self-calibrating non-cartesian parallel imaging. Magnetic Resonance in Medicine 54, 1–8 (2005). <http://doi.wiley.com/10.1002/mrm.20517>`_. This method works extremely well with most data, including cartesian scans. If you are concerned about the quality of your maps then you can save them with ``sense-calib``. Setting the ``--sense-res=X`` option to a large value, e.g. 20 mm, often yields good quality maps.

If your data contains large holes at the center of k-space (e.g. ZTE data) then you will likely want to obtain sensitivity maps from another dataset with ``sense-calib``.

Help! My data is anisotropic!
-----------------------------

Various places in the ``riesling`` code used to make assumptions that both the matrix- and voxel-sizes were isotropic. These assumptions have mostly been removed and anisotropic data should be reconstructed quite happily. Please contact me if it doesn't work. Note that the ``montage`` tool does not understand anisotropic voxels yet so images from that tool may look weird. If you convert to NIFTI the header information should be correct for display.

What if I have Cartesian data?
------------------------------

You need to supply a trajectory, running from -N/2 to (N-1)/2 in each dimension. I then recommend addding either ``--osamp=1.3`` or ``--osamp=1 --kernel=NN``. The latter will effectively switch off the NUFFT functionality and run a plain FFT.

What if I have multiple echoes / frames?
----------------------------------------

The suboptimal way of doing this is to split your non-cartesian data into "volumes" using the time (last) dimension. ``riesling`` will then reconstruct each one separately. This assumes that each echo or frame has exactly the same trajectory.

The better way of doing this is to arrange all of your data into one volume and create a basis using `basis-echoes` or `basis-frames` and pass it to the reconstruction command. ``riesling`` will then split your data as desired but reconstruct it simultaneously. This is very powerful - it allows different trajectories, or parts of the trajectory, to be used for different echoes and for regularization to be applied across all of them (e.g. locally low-rank).

Which regularizer should I pick with ADMM?
------------------------------------------

This depends very much on your data. For straightforward single-image reconstruction I recommend ``--tgv``. If your data scaling is correct, then λ on the order of 1e-2 or 1e-3 should produce good results. If instead you are reconstructing multiple images together, either from a subspace or temporal reconstruction, then Locally Low-Rank works well. Set ``--llr-patch=5 --llr-win=1`` for the highest quality possible, but note this will be slow to run. Increasing the window size will yield a big speed increase but can produce a blocky appearance in the final images.

Should I denoise or regularize?
-------------------------------

Many MRI papers use a regularized reconstruction, i.e. they minimize ``|Ax - y| + λ|f(x)|`` where ``A`` is the encoding/system matrix. While this is mathematically elegant it presents practical problems in algorithm choice for large-scale non-cartesian reconstructions. There are to my knowledge only two general purpose algorithms for solving such problems across typical regularizers - ADMM and PDHG. The adaptive version of ADMM in ``riesling`` is essentially parameter-free, but requires expensive solves for the inner problem. Meanwhile PDHG requires a step-size which is detected by the largest eigenvalue of the encoding matrix, which must be found by power iteration before running the reconstruction. While this eigenvalue will be constant for a particular trajectory, it is still tedious to calculate. Additionally, the accelerated form of PDHG is not compatible with some common regularizers.

An alternative to this approach is to first solve the unregularized reconstruction problem ``|Ax̅ - y|`` and then subsequently solve the denoising problem ``|x - x̅| + λ|f(x)|``. This allows using the efficient LSMR algorithm, which is parameterless and allows preconditioning, for the first step and then PDHG for the second step, which no longer requires the calculation of an eigenvalue for the encoding matrix.