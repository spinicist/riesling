#include "types.hpp"

#include "algo/lsmr.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/compose.hpp"
#include "op/grid.hpp"
#include "op/loop.hpp"
#include "op/multiplex.hpp"
#include "op/reshape.hpp"
#include "precon.hpp"

using namespace rl;

auto MakeGrid(GridOpts &gridOpts, Trajectory const &traj, Index const nC, Index const nSlab, Index const nT, Basis const &basis)
  -> TOps::TOp<Cx, 6, 5>::Ptr
{
  if (gridOpts.vcc) {
    auto grid =
      TOps::Grid<3, true>::Make(traj, gridOpts.ktype.Get(), gridOpts.osamp.Get(), nC, basis, gridOpts.subgridSize.Get());
    auto const ns = grid->ishape;
    auto       reshape =
      std::make_shared<TOps::ReshapeInput<TOps::Grid<3, true>, 5>>(grid, Sz5{ns[0] * ns[1], ns[2], ns[3], ns[4], ns[5]});
    auto loop = TOps::MakeLoop(reshape, nSlab);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(reshape->ishape, nSlab);
    auto slabLoop = TOps::MakeCompose(slabToVol, loop);
    auto timeLoop = TOps::MakeLoop(slabLoop, nT);
    return timeLoop;
  } else {
    auto grid =
      TOps::Grid<3, false>::Make(traj, gridOpts.ktype.Get(), gridOpts.osamp.Get(), nC, basis, gridOpts.subgridSize.Get());
    auto loop = TOps::MakeLoop(grid, nSlab);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(grid->ishape, nSlab);
    auto compose1 = TOps::MakeCompose(slabToVol, loop);
    auto timeLoop = TOps::MakeLoop(compose1, nT);
    return timeLoop;
  }
}

void main_grid(args::Subparser &parser)
{
  CoreOpts   coreOpts(parser);
  GridOpts   gridOpts(parser);
  PreconOpts preOpts(parser);
  LsqOpts    lsqOpts(parser);
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());

  Trajectory traj(reader, reader.readInfo().voxel_size);
  auto const basis = ReadBasis(coreOpts.basisFile.Get());

  auto const  shape = reader.dimensions();
  auto const  nC = shape[0];
  Index const nS = shape[shape.size() - 2];
  auto const  nT = shape[shape.size() - 1];

  auto const A = MakeGrid(gridOpts, traj, nC, nS, nT, basis);

  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeInfo(reader.readInfo());
  traj.write(writer);

  if (fwd) {
    Cx6 const cart = reader.readTensor<Cx6>();
    Cx5       noncart = A->forward(cart);
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
  } else {
    auto const noncart = reader.readTensor<Cx5>();
    traj.checkDims(FirstN<3>(noncart.dimensions()));

    auto const M = MakeKspacePre(traj, nC, nT, basis, preOpts.type.Get(), preOpts.bias.Get());
    LSMR const lsmr{A, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};
    auto       c = lsmr.run(noncart.data());
    writer.writeTensor(HD5::Keys::Data, A->ishape, c.data(), HD5::Dims::Channels);
  }
  Log::Print("Finished {}", parser.GetCommand().Name());
}
