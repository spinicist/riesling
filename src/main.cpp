#include "main_defs.h"
#include <iostream>

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("RIESLING");
  args::Group commands(parser, "COMMANDS");
  args::Command admm(commands, "admm", "ADMM recon", &main_admm);
  args::Command blend(commands, "blend", "Blend basis images", &main_blend);
  args::Command lookup(commands, "lookup", "Basis dictionary lookup", &main_lookup);
  args::Command cg(commands, "cg", "cgSENSE/Iterative recon w/ Töplitz embedding", &main_cg);
  args::Command compress(commands, "compress", "Apply channel compression", &main_compress);
  args::Command espirit(commands, "espirit", "Create SENSE maps with ESPIRiT", &main_espirit);
  args::Command grid(commands, "grid", "Grid from/to non-cartesian to/from cartesian", &main_grid);
  args::Command hdr(commands, "hdr", "Print the header from an HD5 file", &main_hdr);
  args::Command phantom(commands, "phantom", "Construct a digitial phantom", &main_phantom);
  args::Command phantom_basis(commands, "phantom-basis", "Phantom with basis", &main_phantom_basis);
  args::Command plan(commands, "plan", "Plan FFTs", &main_plan);
  args::Command recon(commands, "recon", "Basic non-iterative reconstruction", &main_recon);
  args::Command sdc(commands, "sdc", "Calculate Sample Density Compensation", &main_sdc);
  args::Command sense(commands, "sense", "Create SENSE maps via direct extraction", &main_sense);
  args::Command sim(commands, "sim", "Simulate a basis set", &main_sim);
  args::Command split(commands, "split", "Split data", &main_split);
  args::Command traj(commands, "traj", "Write out the trajectory and PSF", &main_traj);
  args::Command tgv(commands, "tgv", "Iterative TGV regularised recon", &main_tgv);
  args::Command version(commands, "version", "Print version number", &main_version);
  args::Command zinfandel(commands, "zinfandel", "ZINFANDEL k-space filling", &main_zinfandel);
  args::GlobalOptions globals(parser, global_group);

  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help &) {
    fmt::print("{}\n", parser);
    exit(EXIT_SUCCESS);
  } catch (args::Error &e) {
    fmt::print("{}\n", parser);
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), "{}\n", e.what());
    exit(EXIT_FAILURE);
  }

  exit(EXIT_SUCCESS);
}
