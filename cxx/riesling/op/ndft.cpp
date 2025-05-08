#include "inputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/loopify.hpp"
#include "rl/op/ndft.hpp"
#include "rl/sys/threads.hpp"
#include "rl/types.hpp"

using namespace rl;

void main_ndft(args::Subparser &parser)
{
  CoreArgs<3> coreArgs(parser);
  GridArgs<3> gridArgs(parser);
  PreconArgs  preArgs(parser);
  LSMRArgs    lsqOpts(parser);

  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::Flag adj(parser, "", "Apply adjoint operation", {'a', "adj"});
  ParseCommand(parser, coreArgs.iname, coreArgs.oname);

  HD5::Reader reader(coreArgs.iname.Get());
  Info const  info = reader.readInfo();
  auto const  shape = reader.dimensions();
  auto const  basis = LoadBasis(coreArgs.basisFile.Get());
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeInfo(info);

  Trajectory traj(reader, info.voxel_size, coreArgs.matrix.Get());
  if (fwd) {
    auto const cart = reader.readTensor<Cx6>();
    auto const nC = shape[3];
    auto const nS = 1;
    auto const nT = shape[5];
    auto const ndft = TOps::NDFT<3>::Make(traj.matrixForFOV(gridArgs.fov.Get()), traj.points(), nC, basis.get());
    auto const A = Loopify<TOps::NDFT<3>>(ndft, nS, nT);
    auto const noncart = A->forward(cart);
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
  } else {
    auto const noncart = reader.readTensor<Cx5>();
    auto const nC = shape[0];
    auto const nS = shape[3];
    auto const nT = shape[4];
    auto const ndft = TOps::NDFT<3>::Make(traj.matrixForFOV(gridArgs.fov.Get()), traj.points(), nC, basis.get());
    auto const A = Loopify<TOps::NDFT<3>>(ndft, nS, nT);
    if (adj) {
      auto const cart = A->adjoint(noncart);
      writer.writeTensor(HD5::Keys::Data, cart.dimensions(), cart.data(), HD5::Dims::Channels);
    } else {
      auto const M = ndft->M(preArgs.λ.Get(), nS, nT);
      LSMR const lsmr{A, M, nullptr, lsqOpts.Get()};
      auto const c = lsmr.run(CollapseToConstVector(noncart));
      writer.writeTensor(HD5::Keys::Data, A->ishape, c.data(), HD5::Dims::Channels);
    }
  }
}
