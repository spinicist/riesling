#include "types.hpp"

#include "algo/eig.hpp"
#include "cropper.h"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "precond.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

using namespace rl;

int main_eig(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser, "none");
  SENSE::Opts senseOpts(parser);
  args::Flag adj(parser, "ADJ", "Use adjoint system AA'", {"adj"});
  args::ValueFlag<Index> its(parser, "N", "Max iterations (32)", {'i', "max-its"}, 40);
  args::ValueFlag<std::string> pre(parser, "P", "Pre-conditioner (none/kspace/filename)", {"pre"}, "kspace");
  args::Flag recip(parser, "R", "Output reciprocal of eigenvalue", {"recip"});
  args::Flag savevec(parser, "S", "Output the corresponding eigenvector", {"savevec"});
  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  auto A = make_recon(coreOpts, sdcOpts, senseOpts, traj, reader);
  auto P = make_kspace_pre(pre.Get(), A->oshape[0], traj, ReadBasis(coreOpts.basisFile.Get()));

  if (adj) {
    auto const [val, vec] = PowerMethodAdjoint(A, P, its.Get());
    if (savevec) {
      HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "eig"));
      writer.writeTensor("evec", A->ishape, vec.data());
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  } else {
    auto const [val, vec] = PowerMethodForward(A, P, its.Get());
    if (savevec) {
      HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "eig"));
      writer.writeTensor("evec", A->ishape, vec.data());
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  }
  return EXIT_SUCCESS;
}
