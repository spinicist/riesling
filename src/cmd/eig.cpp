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
  args::Flag adj(parser, "ADJ", "Use adjoint system AA'", {"adj"});
  args::Flag pre(parser, "P", "Use k-space preconditioner", {"pre"});
  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  auto recon = make_recon(coreOpts, sdcOpts, senseOpts, traj, false, reader);

  std::optional<Cx4> P = std::nullopt;
  if (pre) {
    auto sc = KSpaceSingle(traj);
    auto const odims = recon->outputDimensions();
    P = std::make_optional<Cx4>(sc.reshape(Sz4{1, odims[1], odims[2], 1}).broadcast(Sz4{odims[0], 1, 1, odims[3]}).cast<Cx>());
  }

  if (adj) {
    auto const [val, vec] = PowerMethodAdjoint(recon, its.Get(), P);
    HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "eig"));
    writer.writeTensor(vec, "evec");
    fmt::print("{}\n", val);
  } else {
    auto const [val, vec] = PowerMethodForward(recon, its.Get(), P);
    HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "eig"));
    writer.writeTensor(vec, "evec");
    fmt::print("{}\n", val);
  }
  return EXIT_SUCCESS;
}
