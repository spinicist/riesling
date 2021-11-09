#include "main_defs.h"
#include <iostream>

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("RIESLING");
  args::Group commands(parser, "COMMANDS");
  args::Command basis_admm(commands, "basis-admm", "ADMM recon with a basis", &main_basis_admm);
  args::Command basis_blend(commands, "basis-blend", "Blend basis images", &main_basis_blend);
  args::Command basis_cg(commands, "basis-cg", "cgSENSE with basis", &main_basis_cg);
  args::Command basis_dict(commands, "basis-dict", "Dictionary lookup", &main_basis_dict);
  args::Command basis_phantom(
    commands, "basis-phantom", "Phantom with a basis", &main_basis_phantom);
  args::Command basis_recon(commands, "basis-recon", "Basis recon with a basis", &main_basis_recon);
  args::Command basis_sim(commands, "basis-sim", "Simulate a basis set", &main_basis_sim);
  args::Command cg(commands, "cg", "cgSENSE/Iterative recon w/ Töplitz embedding", &main_cg);
  args::Command compress(commands, "compress", "Apply channel compression", &main_compress);
  args::Command espirit(commands, "espirit", "Create SENSE maps with ESPIRiT", &main_espirit);
  args::Command grid(commands, "grid", "Grid from/to non-cartesian to/from cartesian", &main_grid);
  args::Command hdr(commands, "hdr", "Print the header from an HD5 file", &main_hdr);
  args::Command phantom(commands, "phantom", "Construct a digitial phantom", &main_phantom);
  args::Command plan(commands, "plan", "Plan FFTs", &main_plan);
  args::Command recon(commands, "recon", "Basic non-iterative reconstruction", &main_recon);
  args::Command sdc(commands, "sdc", "Calculate Sample Density Compensation", &main_sdc);
  args::Command sense(commands, "sense", "Create SENSE maps via direct extraction", &main_sense);
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
