#include "args.hpp"
#include "rl/log/log.hpp"

using namespace rl;

#define COMMAND(PARSER, NM, CMD, DESC)                                                                                         \
  void          main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(PARSER, CMD, DESC, &main_##NM);

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("RIESLING");
  args::GlobalOptions  globals(parser, global_group);

  args::Group recon(parser, "RECON");
  COMMAND(recon, recon_lsq, "recon-lsq", "Least-squares (iterative) recon");
  COMMAND(recon, recon_lsq2, "recon-lsq2", "2D Least-squares (iterative) recon");
  COMMAND(recon, recon_rlsq, "recon-rlsq", "Regularized least-squares recon");
  COMMAND(recon, recon_rlsq2, "recon-rlsq2", "2D Regularized least-squares recon");
  COMMAND(recon, recon_rss, "recon-rss", "NUFFT + Root-Sum-Squares");

  args::Group data(parser, "DATA");
  COMMAND(data, h5, "h5", "Probe an H5 file");
  COMMAND(data, nii, "nii", "Convert images from H5 to nifti");
  COMMAND(data, compress, "compress", "Compress non-cartesian channels");
  COMMAND(data, diff, "diff", "Take the difference of two datasets");
  COMMAND(data, downsamp, "downsamp", "Downsample non-cartesian data");
  COMMAND(data, downsamp2, "downsamp2", "Downsample 2D non-cartesian data");
  COMMAND(data, move, "move", "Shift and rotate data within FOV");
  COMMAND(data, merge, "merge", "Merge non-cartesian data");
  COMMAND(data, norm, "norm", "Calculate the norm of a dataset");
  COMMAND(data, noisify, "noisify", "Add noise to dataset");
  COMMAND(data, real, "real", "Make complex images real valued");
  COMMAND(data, slice_nc, "slice", "Slice non-cartesian data");
  COMMAND(data, slice_img, "slice-img", "Slice images");

  args::Group sense(parser, "SENSE");
  COMMAND(sense, sense_calib, "sense-calib", "Calibrate SENSE kernels");
  COMMAND(sense, sense_calib2, "sense-calib2", "Calibrate 2D SENSE kernels");
  COMMAND(sense, sense_maps, "sense-maps", "Convert SENSE kernels to maps");
  COMMAND(sense, sense_sim, "sense-sim", "Simulate SENSE maps");

  args::Group basis(parser, "BASIS");
  COMMAND(basis, bernstein, "basis-bernstein", "Bernstein Polynomials");
  COMMAND(basis, blend, "basis-blend", "Blend basis images");
  COMMAND(basis, basis_concat, "basis-concat", "Concatenate bases");
  COMMAND(basis, basis_decay, "basis-decay", "Simple T2 decay");
  COMMAND(basis, echoes, "basis-echoes", "Split echoes from sample dimension");
  COMMAND(basis, frames, "basis-frames", "Create a time-frame basis");
  COMMAND(basis, basis_fourier, "basis-fourier", "Basis of Fourier harmonics");
  COMMAND(basis, basis_img, "basis-img", "Basis from image data + SVD");
  COMMAND(basis, navs, "basis-navs", "Basis for navigator images");
  COMMAND(basis, basis_outer, "basis-outer", "Outer product of bases");
  COMMAND(basis, basis_sim, "basis-sim", "Basis from simulations");
  COMMAND(basis, basis_slice, "basis-slice", "Slice a basis");
  COMMAND(basis, basis_svd, "basis-svd", "Basis from simulations + SVD");

  args::Group op(parser, "OP");
  COMMAND(op, fft, "op-fft", "Cartesian FFT of an image");
  COMMAND(op, grad, "op-grad", "Apply grad/div operator");
  COMMAND(op, grid, "op-grid", "Grid from/to non/cartesian");
  COMMAND(op, grid2, "op-grid2", "Grid from/to non/cartesian");
  // COMMAND(op, ndft, "op-ndft", "Apply forward/adjoint NDFT");
  COMMAND(op, nufft, "op-nufft", "Apply forward/adjoint NUFFT");
  COMMAND(op, nufft2, "op-nufft2", "Apply forward/adjoint 2D NUFFT");
  COMMAND(op, pad, "op-pad", "Pad/crop an image");
  COMMAND(op, prox, "op-prox", "Apply Proximal operators");
  COMMAND(op, op_sense, "op-sense", "Channel combine with SENSE");
  COMMAND(op, wavelets, "op-wavelets", "Apply wavelet transform");

  args::Group util(parser, "UTIL");
  // COMMAND(util, autofocus, "autofocus", "Apply Noll's autofocussing");
  COMMAND(util, denoise, "denoise", "Denoise reconstructed images");
  COMMAND(util, eig, "eig", "Calculate largest eigenvalue / vector");
  COMMAND(util, eig2, "eig2", "2D Calculate largest eigenvalue / vector");
  COMMAND(util, filter, "filter", "Apply Tukey filter to image");
  COMMAND(util, log, "log", "Print log to stdout");
  COMMAND(util, mask, "mask", "Generate a mask from an image");
  COMMAND(util, phantom, "phantom", "Make a phantom image");
  COMMAND(util, precon, "precon", "Precompute preconditioning weights");
  COMMAND(util, psf, "psf", "Estimate Point Spread Function");
  COMMAND(util, resamp, "resamp", "Resample images to desired resolution");
  COMMAND(util, rss, "rss", "Take RSS across a dimension");
  COMMAND(util, rovir, "rovir", "Calculate ROVIR compression matrix");
#ifdef BUILD_MONTAGE
  COMMAND(util, montage, "montage", "Make beautiful output images");
#endif
  COMMAND(util, version, "version", "Print version number");

  try {
    parser.ParseCLI(argc, argv);
    Log::End();
  } catch (args::Help &) {
    fmt::print(stderr, "{}\n", parser.Help());
    return EXIT_SUCCESS;
  } catch (args::Error &e) {
    fmt::print(stderr, "{}\n", parser.Help());
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), "{}\n", e.what());
    return EXIT_FAILURE;
  } catch (Log::Failure &f) {
    Log::Fail(f);
    Log::End();
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    Log::Fail(Log::Failure("None", "{}", e.what()));
    Log::End();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
