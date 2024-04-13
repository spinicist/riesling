#include "parse_args.hpp"

using namespace rl;

void main_sense(args::Subparser &parser)
{
#define COMMAND(NM, CMD, DESC)                                                                                                 \
  int           main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(parser, CMD, DESC, &main_##NM);

  COMMAND(sense_calib, "calib", "Create SENSE maps");
  COMMAND(sense_sim, "sim", "Simulate SENSE maps");
}
