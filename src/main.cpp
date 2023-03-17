#include "cmd/defs.h"
#include "fft/fft.hpp"
#include "log.hpp"

using namespace rl;

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("RIESLING");
  args::Group commands(parser, "COMMANDS");
  args::Command admm(commands, "admm", "ADMM recon", &main_admm);
  args::Command blend(commands, "blend", "Blend basis images", &main_blend);
  args::Command cg(commands, "cg", "cgSENSE/Iterative recon w/ Töplitz embedding", &main_cg);
  args::Command compress(commands, "compress", "Apply channel compression", &main_compress);
  args::Command downsamp(commands, "downsamp", "Downsample dataset", &main_downsamp);
  args::Command eig(commands, "eig", "Calculate largest eigenvalue / vector", &main_eig);
  args::Command espirit(commands, "espirit-calib", "Create SENSE maps with ESPIRiT", &main_espirit);
  args::Command filter(commands, "filter", "Apply Tukey filter to image", &main_filter);
  args::Command frames(commands, "frames", "Create a frame basis", &main_frames);
  args::Command grid(commands, "grid", "Grid from/to non-cartesian to/from cartesian", &main_grid);
  args::Command h5(commands, "h5", "Probe an H5 file", &main_h5);
  args::Command lad(commands, "lad", "Least Absolute Deviations", &main_lad);
  args::Command lookup(commands, "lookup", "Basis dictionary lookup", &main_lookup);
  args::Command lsmr(commands, "lsmr", "Iterative recon with LSMR optimizer", &main_lsmr);
  args::Command lsqr(commands, "lsqr", "Iterative recon with LSQR optimizer", &main_lsqr);
  args::Command meta(commands, "meta", "Print meta-data entries", &main_meta);
  args::Command noisify(commands, "noisify", "Add noise to dataset", &main_noisify);
  args::Command nii(commands, "nii", "Convert h5 to nifti", &main_nii);
  args::Command nufft(commands, "nufft", "Apply forward/reverse NUFFT", &main_nufft);
  args::Command pad(commands, "pad", "Pad / crop an image", &main_pad);
  args::Command pdhg(commands, "pdhg", "Primal-Dual Hybrid Gradient", &main_pdhg);
  args::Command phantom(commands, "phantom", "Construct a digitial phantom", &main_phantom);
  args::Command plan(commands, "plan", "Plan FFTs", &main_plan);
  args::Command pre(commands, "precond", "Precompute preconditioning weights", &main_precond);
  args::Command recon(commands, "recon", "Reconstruction with SENSE maps", &main_recon);
  args::Command reg(commands, "reg", "Apply regularization to an image", &main_reg);
  args::Command rss(commands, "rss", "Reconstruction with Root-Sum-Squares channel combination", &main_rss);
  args::Command sdc(commands, "sdc", "Calculate Sample Density Compensation", &main_sdc);
  args::Command sense(commands, "sense", "Apply SENSE operation", &main_sense);
  args::Command sense_calib(commands, "sense-calib", "Create SENSE maps", &main_sense_calib);
  args::Command sense_sim(commands, "sense-sim", "Simulate SENSE maps", &main_sense_sim);
  args::Command sim(commands, "sim", "Simulate a basis set", &main_sim);
  args::Command split(commands, "split", "Split data", &main_split);
  args::Command traj(commands, "traj", "Write out the trajectory and PSF", &main_traj);
  args::Command transform(commands, "transform", "Apply a transform (wavelets / TV)", &main_transform);
  args::Command tgv(commands, "tgv", "Iterative TGV regularised recon", &main_tgv);
  args::Command version(commands, "version", "Print version number", &main_version);
  // args::Command zinfandel(commands, "zinfandel", "ZINFANDEL k-space filling", &main_zinfandel);
  args::GlobalOptions globals(parser, global_group);
  FFT::Start(argv[0]);
  try {
    parser.ParseCLI(argc, argv);
    FFT::End(argv[0]);
    Log::End();
  } catch (args::Help &) {
    fmt::print(stderr, FMT_STRING("{}\n"), parser.Help());
    exit(EXIT_SUCCESS);
  } catch (args::Error &e) {
    fmt::print(stderr, FMT_STRING("{}\n"), parser.Help());
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), FMT_STRING("{}\n"), e.what());
    exit(EXIT_FAILURE);
  } catch (Log::Failure &f) {
    FFT::End(argv[0]);
    Log::End();
    exit(EXIT_FAILURE);
  }

  exit(EXIT_SUCCESS);
}
