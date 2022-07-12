#include "cmd/defs.h"
#include "fft/fft.hpp"
#include "log.h"

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
  args::Command espirit(commands, "espirit-calib", "Create SENSE maps with ESPIRiT", &main_espirit);
  args::Command grid(commands, "grid", "Grid from/to non-cartesian to/from cartesian", &main_grid);
  args::Command hdr(commands, "hdr", "Print the header from an HD5 file", &main_hdr);
  args::Command h5(commands, "h5", "List datasets in an H5 file", &main_h5);
  args::Command lookup(commands, "lookup", "Basis dictionary lookup", &main_lookup);
  args::Command lsmr(commands, "lsmr", "Iterative recon with LSMR optimizer", &main_lsmr);
  args::Command lsqr(commands, "lsqr", "Iterative recon with LSQR optimizer", &main_lsqr);
  args::Command meta(commands, "meta", "Print meta-data entries", &main_meta);
  args::Command nii(commands, "nii", "Convert h5 to nifti", &main_nii);
  args::Command nufft(commands, "nufft", "Apply forward/reverse NUFFT", &main_nufft);
  args::Command pad(commands, "pad", "Pad / crop an image", &main_pad);
  args::Command phantom(commands, "phantom", "Construct a digitial phantom", &main_phantom);
  args::Command plan(commands, "plan", "Plan FFTs", &main_plan);
  args::Command recon(commands, "recon", "Basic non-iterative reconstruction", &main_recon);
  args::Command reg(commands, "reg", "Apply regularization to an image", &main_reg);
  args::Command sdc(commands, "sdc", "Calculate Sample Density Compensation", &main_sdc);
  args::Command sense(commands, "sense", "Apply SENSE operation", &main_sense);
  args::Command sense_calib(commands, "sense-calib", "Create SENSE maps", &main_sense_calib);
  args::Command sim(commands, "sim", "Simulate a basis set", &main_sim);
  args::Command split(commands, "split", "Split data", &main_split);
  args::Command traj(commands, "traj", "Write out the trajectory and PSF", &main_traj);
  args::Command tgv(commands, "tgv", "Iterative TGV regularised recon", &main_tgv);
  args::Command version(commands, "version", "Print version number", &main_version);
  args::Command zinfandel(commands, "zinfandel", "ZINFANDEL k-space filling", &main_zinfandel);
  args::GlobalOptions globals(parser, global_group);
  FFT::Start();
  try {
    parser.ParseCLI(argc, argv);
    FFT::End();
    Log::End();
  } catch (args::Help &) {
    fmt::print(stderr, FMT_STRING("{}\n"), parser.Help());
    exit(EXIT_SUCCESS);
  } catch (args::Error &e) {
    fmt::print(stderr, FMT_STRING("{}\n"), parser.Help());
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), FMT_STRING("{}\n"), e.what());
    exit(EXIT_FAILURE);
  } catch (Log::Failure &f) {
    FFT::End();
    Log::End();
    exit(EXIT_FAILURE);
  }

  exit(EXIT_SUCCESS);
}
