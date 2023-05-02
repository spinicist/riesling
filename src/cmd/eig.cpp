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
  args::ValueFlag<float> preBias(parser, "BIAS", "Pre-conditioner Bias (1)", {"pre-bias", 'b'}, 1.f);
  args::Flag recip(parser, "R", "Output reciprocal of eigenvalue", {"recip"});
  args::Flag savevec(parser, "S", "Output the corresponding eigenvector", {"savevec"});
  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  auto recon = make_recon(coreOpts, sdcOpts, senseOpts, traj, reader);
  auto M = make_pre(pre.Get(), recon->oshape, traj, ReadBasis(coreOpts.basisFile.Get()), preBias.Get());

  if (adj) {
    auto const [val, vec] = PowerMethodAdjoint(recon, M, its.Get());
    if (savevec) {
      HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "eig"));
      writer.writeTensor("evec", recon->ishape, vec.data());
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  } else {
    auto const [val, vec] = PowerMethodForward(recon, M, its.Get());
    if (savevec) {
      HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "eig"));
      writer.writeTensor("evec", recon->ishape, vec.data());
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  }
  return EXIT_SUCCESS;
}
