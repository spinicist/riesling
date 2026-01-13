#include "inputs.hpp"

#include "rl/algo/lsmr.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include "rl/op/compose.hpp"
#include "rl/op/grid.hpp"
#include "rl/op/loop.hpp"
#include "rl/op/reshape.hpp"
#include "rl/precon.hpp"
#include "rl/types.hpp"

using namespace rl;

template <int ND> auto MakeGrid(
  GridOpts<ND> const &gridOpts, TrajectoryN<ND> const &traj, Index const nC, Index const nS, Index const nT, Basis::CPtr basis)
  -> TOps::TOp<6, 5>::Ptr
{
  typename TOps::TOp<ND + 2, 3>::Ptr G = nullptr;
  if (gridOpts.tophat) {
    G = TOps::Grid<ND, TopHat<1>>::Make(gridOpts, traj, nC, basis);
  } else {
    switch (gridOpts.kW) {
      case 4: G = TOps::Grid<ND, ExpSemi<4>>::Make(gridOpts, traj, nC, basis); break;
      case 6: G = TOps::Grid<ND, ExpSemi<6>>::Make(gridOpts, traj, nC, basis); break;
      case 8: G = TOps::Grid<ND, ExpSemi<8>>::Make(gridOpts, traj, nC, basis); break;
      default:
        throw(Log::Failure("Grid", "Kernel width {} not supported", gridOpts.kW));
    }
  }

  if constexpr (ND == 2) {
    auto GS = TOps::MakeLoop<2, 3>(G, nS);
    auto GST = TOps::MakeLoop<5, 4>(GS, nT);
    return GST;
  } else {
    if (nS == 1) {
      auto RG = TOps::MakeReshapeOutput(G, AddBack(G->oshape, 1));
      auto RGT = TOps::MakeLoop<5, 4>(RG, nT);
      return RGT;
    } else {
      throw(Log::Failure("Recon", "Not supported right now"));
    }
  }
}

template <int ND> void run_grid(args::Subparser &parser)
{
  CoreArgs<ND> coreArgs(parser);
  GridArgs<ND> gridArgs(parser);
  PreconArgs   preArgs(parser);
  LSMRArgs     lsqOpts(parser);
  args::Flag   fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::Flag   adj(parser, "", "Apply adjoint operation", {'a', "adj"});

  ParseCommand(parser, coreArgs.iname, coreArgs.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreArgs.iname.Get());

  TrajectoryN<ND> traj(reader, reader.readStruct<Info>(HD5::Keys::Info).voxel_size.head<ND>(), coreArgs.matrix.Get());
  auto const      basis = LoadBasis(coreArgs.basisFile.Get());

  auto const  shape = reader.dimensions();
  auto const  nC = shape[0];
  Index const nS = shape[shape.size() - 2];
  auto const  nT = shape[shape.size() - 1];
  auto const  A = MakeGrid(gridArgs.Get(), traj, nC, nS, nT, basis.get());

  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeStruct(HD5::Keys::Info, reader.readStruct<Info>(HD5::Keys::Info));
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
    auto const M = MakeKSpacePrecon(preArgs.Get(), gridArgs.Get(), traj, basis.get(), nC, Sz2{nS, nT});
    LSMR const lsmr{A, M, nullptr, lsqOpts.Get()};
    auto const c = lsmr.run(CollapseToConstVector(noncart));
    writer.writeTensor(HD5::Keys::Data, A->ishape, c.data(), HD5::Dims::Channels);
  }
  Log::Print(cmd, "Finished");
}

void main_grid(args::Subparser &parser) { run_grid<3>(parser); }
void main_grid2(args::Subparser &parser) { run_grid<2>(parser); }
