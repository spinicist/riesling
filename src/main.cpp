#include "main_defs.h"
#include <iostream>

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("RIESLING");
  args::Group commands(parser, "COMMANDS");
  args::Command cg(commands, "cg", "cgSENSE/Iterative recon w/ Töplitz embedding", &main_cg);
  args::Command cgvar(commands, "cgvar", "cgSENSE with variable preconditioning", &main_cgvar);
  args::Command compress(commands, "compress", "Apply channel compression", &main_compress);
  args::Command ds(commands, "ds", "Direct Summation (NUFT)", &main_ds);
  args::Command grid(commands, "grid", "Grid from/to non-cartesian to/from cartesian", &main_grid);
  args::Command hdr(commands, "hdr", "Print the header from an HD5 file", &main_hdr);
  args::Command phantom(commands, "phantom", "Construct a digitial phantom", &main_phantom);
  args::Command plan(commands, "plan", "Plan FFTs", &main_plan);
  args::Command rss(commands, "rss", "Recon w/ root-sum-squares channel combo", &main_rss);
  args::Command sense(commands, "sense", "Recon w/ self-calibrating sense combo", &main_sense);
  args::Command split(commands, "split", "Split data", &main_split);
  args::Command traj(commands, "traj", "Write out the trajectory and PSF", &main_traj);
  args::Command tgv(commands, "tgv", "Iterative TGV regularised recon", &main_tgv);
  args::Command version(commands, "version", "Print version number", &main_version);
  args::Command zinfandel(commands, "zinfandel", "ZINFANDEL k-space filling", &main_zinfandel);
  args::GlobalOptions globals(parser, global_group);

  try {
    parser.ParseCLI(argc, argv);
  } catch (args::Help &) {
    std::cerr << parser << '\n';
    exit(EXIT_SUCCESS);
  } catch (args::Error &e) {
    std::cerr << parser << '\n' << e.what() << '\n';
    exit(EXIT_FAILURE);
  }

  exit(EXIT_SUCCESS);
}
