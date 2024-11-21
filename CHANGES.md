# v1.01

- The 🏎️ edition. RIESLING is now much faster due to multiple optimizations (judicious use of fast-math, gridding algorithm improvements, better and more widespread threading, and more besides). Many thanks to Martin Reinecke for suggestions.
- Completely new SENSE calibration algorithm based on NLINV. Sensitivities are now estimated and stored as kernels in k-space, and only inflated to full maps when required. The NLINV regularizer is used during calibration to ensure smooth maps.
- New low memory mode `--lowmem`. This performs the SENSE and NUFFT operators sequentially for each channel and accumulates the result. Reduces the memory required for the oversampled grid to a single channel. The speed penalty is acceptable.
- Experimental Direct Virtual Channel Combination `--decant`, i.e. apply SENSE via convolution in k-space not multiplication in image space. This is much slower, but saves even more memory than `--lowmem`.
- Virtual Conjugate Channels `--vcc` for real-valued image reconstruction. Experimental.
- The way FOV cropping and oversampling works has been refined. Now, the `--fov=x,y,z` option refers to the gridding FOV used during the reconstruction, and the oversampling is applied in addition to this. The `--crop-fov=x,y,z` option is applied at the end (default is the nominal FOV). This makes some behind-the-scenes calculations much simpler.
- Implemented the multi-channel version of Frank Ong's preconditioner. This speeds up convergence, but takes some time to calculate itself. Hence the single-channel is still the default. Specify `--precon=multi` if you want to try it.
- Multi-channel TV regularizer.
