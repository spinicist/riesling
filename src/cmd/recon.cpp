#include "parse_args.hpp"

using namespace rl;

void main_recon(args::Subparser &parser)
{
#define COMMAND(NM, CMD, DESC)                                                                                                 \
  int           main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(parser, CMD, DESC, &main_##NM);

  // COMMAND(lad, "lad", "Least Absolute Deviations");
  COMMAND(pdhg, "pdhg", "Primal-Dual Hybrid Gradient");
  COMMAND(pdhg_setup, "pdhg-setup", "Calculate PDHG step sizes");
  COMMAND(recon_lsq, "lsq", "Least-square (iterative) recon");
  COMMAND(recon_rlsq, "rlsq", "Regularized least-squares recon");
  COMMAND(recon_rss, "rss", "Recon with Root-Sum-Squares");
  COMMAND(recon_sense, "sense", "Recon with SENSE");

  parser.Parse();
}
