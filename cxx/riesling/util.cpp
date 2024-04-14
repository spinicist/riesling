#include "parse_args.hpp"

using namespace rl;

void main_util(args::Subparser &parser)
{
#define COMMAND(NM, CMD, DESC)                                                                                                 \
  int           main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(parser, CMD, DESC, &main_##NM);

  COMMAND(autofocus, "autofocus", "Apply Noll's autofocussing");
  COMMAND(denoise, "denoise", "Denoise reconstructed images");
  COMMAND(compress, "compress", "Apply channel compression");
  COMMAND(downsamp, "downsamp", "Downsample dataset");
  COMMAND(eig, "eig", "Calculate largest eigenvalue / vector");
  COMMAND(filter, "filter", "Apply Tukey filter to image");
  COMMAND(precond, "precond", "Precompute preconditioning weights");
  COMMAND(psf, "psf", "Estimate Point Spread Function");
  COMMAND(sdc, "sdc", "Calculate Sample Density Compensation");
  COMMAND(zinfandel, "zinfandel", "ZINFANDEL k-space filling");
}
