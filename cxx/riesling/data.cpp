#include "parse_args.hpp"

using namespace rl;

void main_data(args::Subparser &parser)
{
#define COMMAND(NM, CMD, DESC)                                                                                                 \
  int           main_##NM(args::Subparser &parser);                                                                            \
  args::Command NM(parser, CMD, DESC, &main_##NM);

  COMMAND(h5, "h5", "Probe an H5 file");
  COMMAND(merge, "merge", "Merge non-cartesian data");
  COMMAND(noisify, "noisify", "Add noise to dataset");
  COMMAND(nii, "nii", "Convert h5 to nifti");
  COMMAND(phantom, "phantom", "Make a phantom image");
  COMMAND(slice, "slice", "Slice non-cartesian data");

  parser.Parse();
}
