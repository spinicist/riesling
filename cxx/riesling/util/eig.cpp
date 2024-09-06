#include "types.hpp"

#include "algo/eig.hpp"
#include "inputs.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "precon.hpp"
#include "sense/sense.hpp"
#include "tensors.hpp"
#include "sys/threads.hpp"

using namespace rl;

void main_eig(args::Subparser &parser)
{
  CoreOpts               coreOpts(parser);
  GridOpts               gridOpts(parser);
  PreconOpts             preOpts(parser);
  SENSE::Opts            senseOpts(parser);
  args::Flag             adj(parser, "ADJ", "Use adjoint system AA'", {"adj"});
  args::ValueFlag<Index> its(parser, "N", "Max iterations (32)", {'i', "max-its"}, 40);
  args::Flag             recip(parser, "R", "Output reciprocal of eigenvalue", {"recip"});
  args::Flag             savevec(parser, "S", "Output the corresponding eigenvector", {"savevec"});
  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  auto const  nC = noncart.dimension(0);
  auto const  nS = noncart.dimension(3);
  auto const  nT = noncart.dimension(4);
  auto const basis = LoadBasis(coreOpts.basisFile.Get());
  auto const  A = Recon::SENSE(coreOpts.ndft, gridOpts, senseOpts, traj, nS, nT, basis.get(), noncart);
  auto const  P = MakeKspacePre(traj, nC, nT, basis.get(), preOpts.type.Get(), preOpts.bias.Get());

  if (adj) {
    auto const [val, vec] = PowerMethodAdjoint(A, P, its.Get());
    if (savevec) {
      HD5::Writer writer(coreOpts.oname.Get());
      writer.writeTensor("evec", A->ishape, vec.data(), {"v", "i", "j", "k"});
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  } else {
    auto const [val, vec] = PowerMethodForward(A, P, its.Get());
    if (savevec) {
      HD5::Writer writer(coreOpts.oname.Get());
      writer.writeTensor("evec", A->ishape, vec.data(), {"v", "i", "j", "k"});
    }
    fmt::print("{}\n", recip ? (1.f / val) : val);
  }
}
