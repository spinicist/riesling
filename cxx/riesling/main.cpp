#include "log.hpp"
#include "parse_args.hpp"

using namespace rl;

#define COMMAND(PARSER, NM, CMD, DESC)                                                                                         \
  int           main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(PARSER, CMD, DESC, &main_##NM);

void main_basis(args::Subparser &parser)
{
  COMMAND(parser, basis_fourier, "fourier", "Basis of Fourier harmonics");
  COMMAND(parser, basis_img, "img", "Basis from image data");
  COMMAND(parser, basis_sim, "sim", "Basis from simulations");
  COMMAND(parser, blend, "blend", "Blend basis images");
  COMMAND(parser, frames, "frames", "Create a time-frame basis");
  COMMAND(parser, ipop_basis, "fat", "Fat/Water basis");
  COMMAND(parser, ipop_combine, "fat-post", "Post-process a Fat/Water image");
  COMMAND(parser, lookup, "lookup", "Basis dictionary lookup");

  parser.Parse();
}

void main_data(args::Subparser &parser)
{
  COMMAND(parser, h5, "h5", "Probe an H5 file");
  COMMAND(parser, merge, "merge", "Merge non-cartesian data");
  COMMAND(parser, noisify, "noisify", "Add noise to dataset");
  COMMAND(parser, nii, "nii", "Convert h5 to nifti");
  COMMAND(parser, slice, "slice", "Slice non-cartesian data");

  parser.Parse();
}

void main_op(args::Subparser &parser)
{
  COMMAND(parser, fft, "fft", "Cartesian FFT of an image");
  COMMAND(parser, grad, "grad", "Apply grad/div operator");
  COMMAND(parser, grid, "grid", "Grid from/to non/cartesian");
  COMMAND(parser, ndft, "ndft", "Apply forward/adjoint NDFT");
  COMMAND(parser, nufft, "nufft", "Apply forward/adjoint NUFFT");
  COMMAND(parser, pad, "pad", "Pad/crop an image");
  COMMAND(parser, prox, "prox", "Apply Proximal operators");
  COMMAND(parser, rss, "rss", "Take RSS across first dimension");
  COMMAND(parser, op_sense, "sense", "Channel combine with SENSE");
  COMMAND(parser, wavelets, "wavelets", "Apply wavelet transform");

  parser.Parse();
}

void main_recon(args::Subparser &parser)
{
  // COMMAND(parser, lad, "lad", "Least Absolute Deviations");
  COMMAND(parser, pdhg, "pdhg", "Primal-Dual Hybrid Gradient");
  COMMAND(parser, pdhg_setup, "pdhg-setup", "Calculate PDHG step sizes");
  COMMAND(parser, recon_lsq, "lsq", "Least-square (iterative) recon");
  COMMAND(parser, recon_rlsq, "rlsq", "Regularized least-squares recon");
  COMMAND(parser, recon_rss, "rss", "Recon with Root-Sum-Squares");
  COMMAND(parser, recon_sense, "sense", "Recon with SENSE");

  parser.Parse();
}

void main_sense(args::Subparser &parser)
{
  COMMAND(parser, sense_calib, "calib", "Create SENSE maps");
  COMMAND(parser, sense_sim, "sim", "Simulate SENSE maps");
}

void main_util(args::Subparser &parser)
{
  COMMAND(parser, autofocus, "autofocus", "Apply Noll's autofocussing");
  COMMAND(parser, denoise, "denoise", "Denoise reconstructed images");
  COMMAND(parser, compress, "compress", "Apply channel compression");
  COMMAND(parser, downsamp, "downsamp", "Downsample dataset");
  COMMAND(parser, eig, "eig", "Calculate largest eigenvalue / vector");
  COMMAND(parser, filter, "filter", "Apply Tukey filter to image");
  COMMAND(parser, phantom, "phantom", "Make a phantom image");
  COMMAND(parser, precond, "precond", "Precompute preconditioning weights");
  COMMAND(parser, psf, "psf", "Estimate Point Spread Function");
  COMMAND(parser, sdc, "sdc", "Calculate Sample Density Compensation");
  COMMAND(parser, zinfandel, "zinfandel", "ZINFANDEL k-space filling");
}

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("RIESLING");
  args::Group          commands(parser, "COMMANDS");

  args::Command basis(commands, "basis", "Create a subspace basis", &main_basis);
  args::Command data(commands, "data", "Manipulate riesling files", &main_data);
#ifdef BUILD_MONTAGE
  COMMAND(commands, montage, "montage", "Make beautiful output images");
#endif
  args::Command op(commands, "op", "Linear Operators", &main_op);
  args::Command recon(commands, "recon", "Reconstruction", &main_recon);
  args::Command sense(commands, "sense", "Sensitivity maps", &main_sense);
  args::Command util(commands, "util", "Utilities", &main_util);
  COMMAND(commands, version, "version", "Print version number");
  args::GlobalOptions globals(parser, global_group);
  try {
    parser.ParseCLI(argc, argv);
    Log::End();
  } catch (args::Help &) {
    fmt::print(stderr, "{}\n", parser.Help());
    exit(EXIT_SUCCESS);
  } catch (args::Error &e) {
    fmt::print(stderr, "{}\n", parser.Help());
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), "{}\n", e.what());
    exit(EXIT_FAILURE);
  } catch (Log::Failure &f) {
    Log::End();
    exit(EXIT_FAILURE);
  }

  exit(EXIT_SUCCESS);
}
