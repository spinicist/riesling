#include "types.hpp"

#include "algo/eig.hpp"
#include "cropper.h"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precond.hpp"
#include "sdc.hpp"
#include "sense.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

using namespace rl;

int main_eig(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser);
  SENSE::Opts senseOpts(parser);
  args::ValueFlag<Index> its(parser, "N", "Max iterations (32)", {'i', "max-its"}, 32);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  auto recon = make_recon(coreOpts, sdcOpts, senseOpts, traj, false, reader);

  auto const [val, vec] = PowerMethod(recon, its.Get());

  HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "eig"));
  writer.writeTensor(vec, "evec");
  fmt::print("{}\n", val);
  return EXIT_SUCCESS;
}
