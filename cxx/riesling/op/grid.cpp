#include "inputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/op/compose.hpp"
#include "rl/op/grid.hpp"
#include "rl/op/loop.hpp"
#include "rl/op/multiplex.hpp"
#include "rl/op/reshape.hpp"
#include "rl/precon.hpp"
#include "rl/types.hpp"

using namespace rl;

auto MakeGrid(
  GridOpts<3> const &gridOpts, Trajectory const &traj, Index const nC, Index const nS, Index const nT, Basis::CPtr basis)
  -> TOps::TOp<Cx, 6, 5>::Ptr
{
  auto grid = TOps::Grid<3>::Make(gridOpts, traj, nC, basis);
  if (nS == 1) {
    auto rout = TOps::MakeReshapeOutput(grid, AddBack(grid->oshape, 1));
    auto timeLoop = TOps::MakeLoop<4, 4>(rout, nT);
    return timeLoop;
  } else {
    auto loop = TOps::MakeLoop<3, 3>(grid, nS);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(grid->ishape, nS);
    auto compose1 = TOps::MakeCompose(slabToVol, loop);
    auto timeLoop = TOps::MakeLoop<4, 4>(compose1, nT);
    return timeLoop;
  }
}

void main_grid(args::Subparser &parser)
{
  CoreArgs<3> coreArgs(parser);
  GridArgs<3> gridArgs(parser);
  PreconArgs  preArgs(parser);
  LSMRArgs    lsqOpts(parser);
  args::Flag  fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::Flag  adj(parser, "", "Apply adjoint operation", {'a', "adj"});

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());

  Trajectory traj(reader, reader.readInfo().voxel_size, coreArgs.matrix.Get());
  auto const basis = LoadBasis(coreArgs.basisFile.Get());

  auto const  shape = reader.dimensions();
  auto const  nC = shape[0];
  Index const nS = shape[shape.size() - 2];
  auto const  nT = shape[shape.size() - 1];
  auto const  A = MakeGrid(gridArgs.Get(), traj, nC, nS, nT, basis.get());

  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeInfo(reader.readInfo());
  traj.write(writer);

  if (fwd) {
    auto const cart = reader.readTensor<Cx6>();
    auto const noncart = A->forward(cart);
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
  } else if (adj) {
    auto const noncart = reader.readTensor<Cx5>();
    auto const cart = A->adjoint(noncart);
    writer.writeTensor(HD5::Keys::Data, cart.dimensions(), cart.data(), HD5::Dims::Channels);
  } else {
    auto const noncart = reader.readTensor<Cx5>();
    traj.checkDims(FirstN<3>(noncart.dimensions()));
    auto const M = MakeKSpacePrecon(preArgs.Get(), gridArgs.Get(), traj, nC, nS, nT);
    LSMR const lsmr{A, M, nullptr, lsqOpts.Get()};
    auto const c = lsmr.run(CollapseToConstVector(noncart));
    writer.writeTensor(HD5::Keys::Data, A->ishape, c.data(), HD5::Dims::Channels);
  }
  Log::Print(cmd, "Finished");
}
