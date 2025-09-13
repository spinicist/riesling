## Changelog

# v1.03

- The 🐛 edition. Sadly some bugs to do with data ordering and subspace recon had slipped into v1.02. These have been fixed.
- The basis and channel dimensions in image space have been swapped, channels is now outermost, giving a further performance boost. The dimensions in the input data remain the same. Hence you do not need to reconvert any data to the RIESLING format, but you will need to regenerate sense maps created with v1.02.
- The PDHG algorithm implementation has been much improved. The regularizer operators are now all scaled to have a maximum eigenvalue of 1. You will still need to calculate the maximum eigenvalue of your encoding operator.
- New regularizers: `--tv2` for second-order TV (normal TV + the Laplacian) which is fast alternative to TGV, `--l1i` to apply L1 regularization to the imaginary part only (i.e. the image should be real-valued).
- A working implementation of ROVir.
- An option to change the gridding kernel width has been added back.

# v1.02

- The 🫨 edition. RIESLING now includes:
- A solid implementation of the Primal Dual Hybrid Gradient algorithm. This gives much faster denoising and regularized reconstructions as long as you calculate an eigenvalue up front (see docs).
- 2D versions of all reconstruction tools and most operators. These have the same command names with a `2` suffix. When using these, you should supply a 2D trajectory (no kz co-ordinate). This will then use a true 2D NUFFT.
- Off-resonance correction. If you supply an off-resonance map in the input .h5 and use the `--tacq` and `--Nt` arguments RIESLING will perform a time-segmented reconstruction. Useful for long acquisition spiral imaging.
- An implementation of MERLIN. This is contained in a separate executable that is not compiled by default, enable BUILD_MERLIN to generate it.
- Note that the image space data order has changed, the spatial dimensions are now inner-most (i.e. channels and basis dimension have moved after the spatial dimensions). This simplifies some RIESLING internals.

# v1.01

- The 🏎️ edition. RIESLING is now much faster due to multiple optimizations (judicious use of fast-math, gridding algorithm improvements, better and more widespread threading, and more besides). Many thanks to Martin Reinecke for suggestions.
- Completely new SENSE calibration algorithm based on NLINV. Sensitivities are now estimated and stored as kernels in k-space, and only inflated to full maps when required. The NLINV regularizer is used during calibration to ensure smooth maps.
- New low memory mode `--lowmem`. This performs the SENSE and NUFFT operators sequentially for each channel and accumulates the result. Reduces the memory required for the oversampled grid to a single channel. The speed penalty is acceptable.
- Experimental Direct Virtual Channel Combination `--decant`, i.e. apply SENSE via convolution in k-space not multiplication in image space. This is much slower, but saves even more memory than `--lowmem`.
- Virtual Conjugate Channels `--vcc` for real-valued image reconstruction. Experimental.
- The way FOV cropping and oversampling works has been refined. Now, the `--fov=x,y,z` option refers to the gridding FOV used during the reconstruction, and the oversampling is applied in addition to this. The `--crop-fov=x,y,z` option is applied at the end (default is the nominal FOV). This makes some behind-the-scenes calculations much simpler.
- Implemented the multi-channel version of Frank Ong's preconditioner. This speeds up convergence, but takes some time to calculate itself. Hence the single-channel is still the default. Specify `--precon=multi` if you want to try it.
- Multi-channel TV regularizer.

# v1.00

- I now consider RIESLING useable enough to declare version 1 🥂.
- _The trajectory scaling now matches BART (-M/2 to M/2 where M is matrix size)._
- The interface has been much simplified - input .h5 files are expected to have a dataset named "data" and commands take both input and output filenames. This removes the need to remember the specific dataset names and suffixes added by commands.
- Useful commands have been renamed and reorganized to aid discoverability. The most important are `recon-lsq` and `recon-rlsq`, which are now named by the problem they solve (Least-Squares and Regularized Least-Squares respectively) instead of the particular algorithms used (previously they were `lsmr` and `admm`).
- The new `montage` tool can create pretty montages from RIESLING datasets. These can be saved as PNGs or displayed in the terminal if you use KiTTY. Sadly, due to dependency issues, this command is not included in the Github downloads and you will need to compile on your local machine if you want to use it.
- Some commands have been removed, in particular `cg`. The remaining commands have many advantages and maintaing all the different algorithms was costing considerable time.
- Behind the scenes, this version can be considered the "preconditioned least-squares everywhere" release. The operator commands now use a few iterations of preconditioned LSMR to calculute the inverse NUFFT etc. in preference to Density Compensation. This approach was already used for SENSE calibration and I consider it the superior approach over Density Compensation. A discussion on this is welcome at ISMRM, especially if you provide a glass of riesling.
- RIESLING now uses the DUCC FFT library instead of FFTW. This has comparable performance, no planning, and a much better multi-threading implementation.
- New and improved regularizers including wavelets and TGV on the L2-norm of multi-channel images.

# v0.12

- A much improved ADMM implementation, including the residual balancing scheme from Wohlberg 2017. This shows much more robust convergence behaviour. The default number of inner iterations is now only 1, giving improved speed.
- Implemented a version of PDHG that supports multiple regularizers. However, I prefer the improved ADMM implementation.
- Implemented the NDFT (Non-uniform Discrete Fourier Transform). This is incredibly slow compared to the NUFFT, but revealed some small innacuracies in the NUFFT implementation. These have now been fixed.

# v0.11

- Fixed many bugs that crept into v0.10.
- Functionality to write out residual k-space/images
- Small but important tweaks to how the ADMM algorithm works including more sensible defaults.
- Added a through-time TV regularizer option for ADMM.
- Added a tool to calcuate a basis set from temporal images.

# v0.10

- The `admm` command now supports TGV regularization. The separate `tgv` command, which used the form of PDHG in the original paper, has hence been retired. The `admm` version is superior as it properly supports preconditioning.
- The log is now saved into the output .h5 file for late reference.
- Iterative algorithms now scale the data. This makes regularization parameters (λ) behave sensibly. The default is to calculate a scaling factor by perfomring an adjoint NUFFT, using Otsu's method to identify foreground signal, and then taking the median value. If `--scale=bart` is used then the BART scaling of 90th percentile is used. A pre-calculated factor can also be specified with `--scale=X`.
- Improvements to loading/saving data from Python thanks to Martin Krämer.
- A basis can now be specified for several plotting functions.
- The FFT wisdom is now saved to a path that includes the hostname and executable name. This gives more consistent behaviour in networked environments with hetergeneous server hardware.
- Added a maximum entropy regularizer.

# Before

The changelog was only started for version 0.10.