#include "fft/fft.hpp"
#include "log.hpp"
#include "parse_args.hpp"

using namespace rl;

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("RIESLING");
  args::Group          commands(parser, "COMMANDS");

#define COMMAND(NM, CMD, DESC)                                                                                                 \
  int           main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(commands, CMD, DESC, &main_##NM);

  COMMAND(admm, "admm", "ADMM recon");
  COMMAND(basis_img, "basis-img", "Basis from image data");
  COMMAND(basis_sim, "basis-sim", "Basis from simulations");
  COMMAND(blend, "blend", "Blend basis images");
  COMMAND(cg, "cg", "Recon with Conjugate Gradients");
  COMMAND(compress, "compress", "Apply channel compression");
  COMMAND(denoise, "denoise", "Denoise reconstructed images");
  COMMAND(downsamp, "downsamp", "Downsample dataset");
  COMMAND(eig, "eig", "Calculate largest eigenvalue / vector");
  COMMAND(filter, "filter", "Apply Tukey filter to image");
  COMMAND(frames, "frames", "Create a frame basis");
  COMMAND(grid, "grid", "Grid from/to non/cartesian");
  COMMAND(h5, "h5", "Probe an H5 file");
  // COMMAND(lad, "lad", "Least Absolute Deviations");
  COMMAND(lookup, "lookup", "Basis dictionary lookup");
  COMMAND(lsmr, "lsmr", "Recon with LSMR optimizer");
  COMMAND(lsqr, "lsqr", "Recon with LSQR optimizer");
  COMMAND(noisify, "noisify", "Add noise to dataset");
  COMMAND(nii, "nii", "Convert h5 to nifti");
  COMMAND(ndft, "ndft", "Apply forward/adjoint NDFT");
  COMMAND(nufft, "nufft", "Apply forward/adjoint NUFFT");
  COMMAND(pad, "pad", "Pad/crop an image");
  COMMAND(pdhg, "pdhg", "Primal-Dual Hybrid Gradient");
  COMMAND(pdhg_setup, "pdhg-setup", "Calculate PDHG step sizes");
  COMMAND(phantom, "phantom", "Construct a digitial phantom");
  COMMAND(plan, "plan", "Plan FFTs");
  COMMAND(precond, "precond", "Precompute preconditioning weights");
  COMMAND(prox, "prox", "Apply proximal operator");
  COMMAND(recon, "recon", "Recon with SENSE maps");
  COMMAND(rss, "rss", "Recon with Root-Sum-Squares");
  COMMAND(sdc, "sdc", "Calculate Sample Density Compensation");
  COMMAND(sense, "sense", "Apply SENSE operation");
  COMMAND(sense_calib, "sense-calib", "Create SENSE maps");
  COMMAND(sense_sim, "sense-sim", "Simulate SENSE maps");
  COMMAND(slice, "slice", "Slice non-cartesian data");
  COMMAND(traj, "traj", "Write out the trajectory and PSF");
  COMMAND(transform, "transform", "Apply a transform (wavelets/TV)");
  COMMAND(version, "version", "Print version number");
  COMMAND(zinfandel, "zinfandel", "ZINFANDEL k-space filling");
  args::GlobalOptions globals(parser, global_group);
  FFT::Start(argv[0]);
  try {
    parser.ParseCLI(argc, argv);
    FFT::End(argv[0]);
    Log::End();
  } catch (args::Help &) {
    fmt::print(stderr, "{}\n", parser.Help());
    exit(EXIT_SUCCESS);
  } catch (args::Error &e) {
    fmt::print(stderr, "{}\n", parser.Help());
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), "{}\n", e.what());
    exit(EXIT_FAILURE);
  } catch (Log::Failure &f) {
    FFT::End(argv[0]);
    Log::End();
    exit(EXIT_FAILURE);
  }

  exit(EXIT_SUCCESS);
}
