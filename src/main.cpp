#include "fftw3.h"
#include "main_defs.h"
#include <iostream>

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser parser("RIESLING");
  args::Group commands(parser, "COMMANDS");
  args::Command hdr(commands, "hdr", "Print the header from an HD5 file", &main_hdr);
  args::Command phantom(commands, "phantom", "Construct a digitial phantom", &main_phantom);
  args::Command zinfandel(commands, "zinfandel", "ZINFANDEL k-space filling", &main_zinfandel);
  args::Command kspace(commands, "kspace", "Output radial k-space for viewing", &main_kspace);
  args::Command traj(commands, "traj", "Write out the trajectory and PSF", &main_traj);
  args::Command rss(commands, "rss", "Basic root-sum-squares recon", &main_rss);
  args::Command sense(commands, "sense", "Calculate sensitivity maps", &main_sense);
  args::Command tgv(commands, "tgv", "Iterative TGV recon", &main_tgv);
  args::Command tp(commands, "toe", "Iterative Toeplitz recon", &main_toeplitz);
  args::Command ds(commands, "ds", "Direct Summation (NUFT)", &main_ds);
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
