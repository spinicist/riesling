## Changelog

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