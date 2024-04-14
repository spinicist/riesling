#include "parse_args.hpp"

void main_op(args::Subparser &parser)
{
#define COMMAND(NM, CMD, DESC)                                                                                                 \
  int           main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(parser, CMD, DESC, &main_##NM);

  COMMAND(fft, "fft", "Cartesian FFT of an image");
  COMMAND(grad, "grad", "Apply grad/div operator");
  COMMAND(grid, "grid", "Grid from/to non/cartesian");
  COMMAND(ndft, "ndft", "Apply forward/adjoint NDFT");
  COMMAND(nufft, "nufft", "Apply forward/adjoint NUFFT");
  COMMAND(pad, "pad", "Pad/crop an image");
  COMMAND(rss, "rss", "Take RSS across first dimension");
  COMMAND(op_sense, "sense", "Channel combine with SENSE");
  COMMAND(wavelets, "wavelets", "Apply wavelet transform");

  parser.Parse();
}
