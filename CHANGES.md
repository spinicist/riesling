## Changelog

# v0.10

- Iterative algorithms now scale the data. This makes regularization parameters (λ) behave sensibly. The default is to calculate a scaling factor by perfomring an adjoint NUFFT, using Otsu's method to identify foreground signal, and then taking the median value. If `--scale=bart` is used then the BART scaling of 90th percentile is used. A pre-calculated factor can also be specified with `--scale=X`.
- Improvements to loading/saving data from Python thanks to Martin Krämer. Basis vectors can now be specified for several plotting functions.
- The FFT wisdom is now saved to a path that includes the hostname and executable name. This gives more consistent behaviour in networked environments with hetergeneous server hardware.
- Added a maximum entropy regularizer and Least Absolute Deviations solver.

# Before

The changelog was only started for version 0.10.