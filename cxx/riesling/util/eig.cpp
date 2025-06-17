#include "inputs.hpp"
#include "outputs.hpp"

#include "rl/algo/eig.hpp"
#include "rl/algo/lsmr.hpp"
#include "rl/log/log.hpp"
#include "rl/op/pad.hpp"
#include "rl/op/recon.hpp"
#include "rl/precon.hpp"
#include "rl/scaling.hpp"
#include "rl/sense/sense.hpp"
#include "rl/types.hpp"

using namespace rl;

template <int ND> void run_eig(args::Subparser &parser)
{
  CoreArgs<ND>           coreArgs(parser);
  GridArgs<ND>           gridArgs(parser);
  PreconArgs             preArgs(parser);
  ReconArgs              reconArgs(parser);
  SENSEArgs<ND>          senseArgs(parser);
  f0Args                 f0Args(parser);
  args::Flag             adj(parser, "ADJ", "Use adjoint system AA'", {"adj"});
  args::ValueFlag<Index> its(parser, "N", "Max iterations (32)", {'i', "max-its"}, 40);
  args::Flag             recip(parser, "R", "Output reciprocal of eigenvalue", {"recip"});
  args::Flag             savevec(parser, "S", "Output the corresponding eigenvector", {"savevec"});
  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const      cmd = parser.GetCommand().Name();
  HD5::Reader     reader(coreArgs.iname.Get());
  Info const      info = reader.readStruct<Info>(HD5::Keys::Info);
  TrajectoryN<ND> traj(reader, info.voxel_size.head<ND>(), coreArgs.matrix.Get());
  auto            noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));

  auto const basis = LoadBasis(coreArgs.basisFile.Get());
  auto const R = f0Args.Nτ ? Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), senseArgs.Get(), traj, f0Args.Get(), noncart,
                                   reader.readTensor<Re3>("f0map"))
                           : Recon(reconArgs.Get(), preArgs.Get(), gridArgs.Get(), senseArgs.Get(), traj, basis.get(), noncart);

  if (adj) {
    auto const [val, vec] = PowerMethodAdjoint(R.A, R.M, its.Get());
    if (savevec) {
      HD5::Writer writer(coreArgs.oname.Get());
      writer.writeTensor("evec", R.A->ishape, vec.data(), {"v", "i", "j", "k"});
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  } else {
    auto const [val, vec] = PowerMethodForward(R.A, R.M, its.Get());
    if (savevec) {
      HD5::Writer writer(coreArgs.oname.Get());
      writer.writeTensor("evec", R.A->ishape, vec.data(), {"v", "i", "j", "k"});
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  }
}

void main_eig(args::Subparser &parser) { run_eig<3>(parser); }
void main_eig2(args::Subparser &parser) { run_eig<2>(parser); }
