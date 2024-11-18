#include "types.hpp"

#include "algo/eig.hpp"
#include "inputs.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "precon.hpp"
#include "sense/sense.hpp"
#include "sys/threads.hpp"
#include "tensors.hpp"

using namespace rl;

void main_eig(args::Subparser &parser)
{
  CoreArgs               coreArgs(parser);
  GridArgs<3>            gridArgs(parser);
  PreconArgs             preArgs(parser);
  ReconArgs              reconArgs(parser);
  SENSE::Opts            senseOpts(parser);
  args::Flag             adj(parser, "ADJ", "Use adjoint system AA'", {"adj"});
  args::ValueFlag<Index> its(parser, "N", "Max iterations (32)", {'i', "max-its"}, 40);
  args::Flag             recip(parser, "R", "Output reciprocal of eigenvalue", {"recip"});
  args::Flag             savevec(parser, "S", "Output the corresponding eigenvector", {"savevec"});
  ParseCommand(parser, coreArgs.iname, coreArgs.oname);

  HD5::Reader reader(coreArgs.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  auto const  nC = noncart.dimension(0);
  auto const  nS = noncart.dimension(3);
  auto const  nT = noncart.dimension(4);
  auto const  basis = LoadBasis(coreArgs.basisFile.Get());
  auto const  A = Recon::Choose(reconArgs.Get(), gridArgs.Get(), senseOpts, traj, basis.get(), noncart);
  auto const  P = MakeKSpaceSingle(preArgs.Get(), gridArgs.Get(), traj, nC, nS, nT, basis.get());

  if (adj) {
    auto const [val, vec] = PowerMethodAdjoint(A, P, its.Get());
    if (savevec) {
      HD5::Writer writer(coreArgs.oname.Get());
      writer.writeTensor("evec", A->ishape, vec.data(), {"v", "i", "j", "k"});
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  } else {
    auto const [val, vec] = PowerMethodForward(A, P, its.Get());
    if (savevec) {
      HD5::Writer writer(coreArgs.oname.Get());
      writer.writeTensor("evec", A->ishape, vec.data(), {"v", "i", "j", "k"});
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  }
}
