#include "types.hpp"

#include "algo/lsmr.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/compose.hpp"
#include "op/grid.hpp"
#include "op/loop.hpp"
#include "op/multiplex.hpp"
#include "op/reshape.hpp"
#include "parse_args.hpp"
#include "precon.hpp"

using namespace rl;

auto MakeGrid(GridOpts &gridOpts, Trajectory const &traj, Index const nC, Index const nSlab, Basis const &basis)
  -> TOps::TOp<Cx, 5, 4>::Ptr
{
  if (gridOpts.vcc) {
    auto grid =
      TOps::Grid<3, true>::Make(traj, gridOpts.ktype.Get(), gridOpts.osamp.Get(), nC, basis, gridOpts.subgridSize.Get());
    auto const ns = grid->ishape;
    auto       reshape =
      std::make_shared<TOps::ReshapeInput<TOps::Grid<3, true>, 5>>(grid, Sz5{ns[0] * ns[1], ns[2], ns[3], ns[4], ns[5]});
    auto loop = std::make_shared<TOps::Loop<TOps::TOp<Cx, 5, 3>>>(reshape, nSlab);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(reshape->ishape, nSlab);
    auto compose2 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
    return compose2;
  } else {
    auto grid =
      TOps::Grid<3, false>::Make(traj, gridOpts.ktype.Get(), gridOpts.osamp.Get(), nC, basis, gridOpts.subgridSize.Get());
    auto loop = std::make_shared<TOps::Loop<TOps::Grid<3, false>>>(grid, nSlab);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(grid->ishape, nSlab);
    auto compose1 = std::make_shared<decltype(TOps::Compose(slabToVol, loop))>(slabToVol, loop);
    return compose1;
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
  auto const  nV = shape[shape.size() - 1];

  auto const A = MakeGrid(gridOpts, traj, nC, nS, basis);

  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeInfo(reader.readInfo());
  traj.write(writer);

  if (fwd) {
    Cx6 const cart = reader.readTensor<Cx6>();
    Cx5       noncart(AddBack(A->oshape, nV));
    for (Index iv = 0; iv < nV; iv++) {
      noncart.chip<4>(iv).chip<3>(0).device(Threads::GlobalDevice()) = A->forward(CChipMap(cart, iv));
    }
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
  } else {
    auto const noncart = reader.readTensor<Cx5>();
    traj.checkDims(FirstN<3>(noncart.dimensions()));

    auto const M = make_kspace_pre(traj, nC, basis, gridOpts.vcc, preOpts.type.Get(), preOpts.bias.Get());
    LSMR const lsmr{A, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};

    Cx6 cart(AddBack(A->ishape, nV));
    for (Index iv = 0; iv < nV; iv++) {
      cart.chip<5>(iv).device(Threads::GlobalDevice()) = Tensorfy(lsmr.run(&noncart(0, 0, 0, 0, iv)), A->ishape);
    }
    writer.writeTensor(HD5::Keys::Data, cart.dimensions(), cart.data(), HD5::Dims::Channels);
  }
  Log::Print("Finished {}", parser.GetCommand().Name());
}
