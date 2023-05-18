Frequently Asked Questions
==========================

Understandly, users have questions. Here are some answers.

#. `What is the trajectory scaling?`_
#. `When do I need Sample Density Compensation?`_
#. `Do I need to supply sensitivity maps?`_
#. `What if I have Cartesian data?`_
#. `Which regularizer should I pick?`_

What is the trajectory scaling?
-------------------------------

The extent of the non-cartesian k-space is scaled as -0.5 to 0.5 in each dimension. The Cartesian grid indices are calculated through a combination of the trajectory, desired output matrix size and the oversampling factor. Note that the cartesian indices run from -N/2 to N/2-1, which is determined by the definition of the FFT. If you have a truly symmetric non-cartesian trajectory then one k-space sample will be dropped at the positive edge of each dimension. This is correct behaviour for the FFT.

When do I need Sample Density Compensation?
-------------------------------------------

Only when using the ``rss``, ``sense`` or ``cg`` commands. If you don't supply SDC factors to those commands they will be calculated automatically. The other reconstruction commands (``lsmr`` and ``admm``) use preconditioning by default instead of SDC. Calculating the preconditioning is much faster than calculating SDC, but you can still save time by calculating the preconditioning weights up front with the ``precond`` command. See `F. Ong, M. Uecker, and M. Lustig, ‘Accelerating Non-Cartesian MRI Reconstruction Convergence Using k-Space Preconditioning’, IEEE Trans. Med. Imaging, vol. 39, no. 5, pp. 1646–1654, May 2020<https://ieeexplore.ieee.org/document/8906069/>`_.

Do I need to supply sensitivity maps?
-------------------------------------

For most data you do not. ``riesling`` will generate the maps by heavily filtering the central region of k-space, performing a NUFFT and then normalizing the resulting channel images by the root-sum-of squares. See `Yeh, E. N. et al. Inherently self-calibrating non-cartesian parallel imaging. Magnetic Resonance in Medicine 54, 1–8 (2005). <http://doi.wiley.com/10.1002/mrm.20517>`_. This method works extremely well with most data, including cartesian scans. If you are concerned about the quality of your maps then you can save them with ``sense-calib``. Setting the ``--sense-res=X`` option to a large value, e.g. 20 mm, often yields good quality maps. ``riesling`` also contains an implementation of ESPiRIT.

If your data contains large holes at the center of k-space (e.g. ZTE data) then you will likely want to obtain sensitivity maps from another dataset with ``sense-calib``.

What if I have Cartesian data?
------------------------------

The default gridding options in ``riesling`` were picked for non-cartesian data and likely will not work with a cartesian dataset. In particular, a twice over-sampled grid is used and the FOV during iterations (determined by `--sense-fov``) is expanded to 256 mm on the basis that most non-cartesian acquisitions oversample the center of k-space and so in practice acquire signal from outside the nominal FOV. These settings will cause artefacts during an iterative recon with cartesian data.

Adding ``--sense-fov=-1`` will instead crop the sensitivity maps tightly the FOV. I then recommend addding either ``--osamp=1.3`` or ``--osamp=1 --kernel=NN``. The latter will effectively switch off the NUFFT functionality and run a plain FFT instead.

Which regularizer should I pick with ADMM?
------------------------------------------

This depends very much on your data. For straightforward single-image reconstruction I recommend ``--tgv``. If your data scaling is correct, then λ on the order of 1e-2 or 1e-3 should produce good results. If instead you are reconstructing multiple images together, either from a subspace or temporal reconstruction, then Locally Low-Rank works well. Set ``--llr-patch=5 --llr-win=1`` for the highest quality possible, but note this will be slow to run. Increasing the window size will yield a big speed increase but can produce a blocky appearance in the final images.
