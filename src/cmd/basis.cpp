#include "parse_args.hpp"

using namespace rl;

void main_basis(args::Subparser &parser)
{
#define COMMAND(NM, CMD, DESC)                                                                                                 \
  int           main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(parser, CMD, DESC, &main_##NM);

  COMMAND(basis_fourier, "fourier", "Basis of Fourier harmonics");
  COMMAND(basis_img, "img", "Basis from image data");
  COMMAND(basis_sim, "sim", "Basis from simulations");
  COMMAND(blend, "blend", "Blend basis images");
  COMMAND(frames, "frames", "Create a time-frame basis");
  COMMAND(ipop_basis, "fat", "Fat/Water basis");
  COMMAND(ipop_combine, "fat-post", "Post-process a Fat/Water image");
  COMMAND(lookup, "lookup", "Basis dictionary lookup");

  parser.Parse();
}
