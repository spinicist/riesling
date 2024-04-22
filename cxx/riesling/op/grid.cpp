#include "types.hpp"

#include "algo/lsmr.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grid.hpp"
#include "parse_args.hpp"
#include "precond.hpp"

using namespace rl;

void main_grid(args::Subparser &parser)
{
  CoreOpts               coreOpts(parser);
  GridOpts               gridOpts(parser);
  PrecondOpts            preOpts(parser);
  LsqOpts                lsqOpts(parser);
  args::Flag             fwd(parser, "", "Apply forward operation", {'f', "fwd"});

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());

  Trajectory traj(reader, reader.readInfo().voxel_size);
  auto const basis = ReadBasis(coreOpts.basisFile.Get());

  auto const shape = reader.dimensions();
  auto const nC = shape[0];
  auto const nV = shape[shape.size() - 1];

  auto const A = Grid<Cx, 3>::Make(traj, gridOpts.ktype.Get(), gridOpts.osamp.Get(), nC, basis);

  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeInfo(reader.readInfo());
  traj.write(writer);

  if (fwd) {
    Cx6 const cart = reader.readTensor<Cx6>();
    Cx5       noncart(AddBack(A->oshape, 1, nV));
    for (Index iv = 0; iv < nV; iv++) {
      noncart.chip<4>(iv).chip<3>(0).device(Threads::GlobalDevice()) = A->forward(CChipMap(cart, iv));
    }
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
  } else {
    auto const noncart = reader.readTensor<Cx5>();
    traj.checkDims(FirstN<3>(noncart.dimensions()));

    auto const M = make_kspace_pre(traj, nC, basis, preOpts.type.Get(), preOpts.bias.Get());
    LSMR const lsmr{A, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};

    Cx6 cart(AddBack(A->ishape, nV));
    for (Index iv = 0; iv < nV; iv++) {
      cart.chip<5>(iv).device(Threads::GlobalDevice()) = Tensorfy(lsmr.run(&noncart(0, 0, 0, 0, iv)), A->ishape);
    }
    writer.writeTensor(HD5::Keys::Data, cart.dimensions(), cart.data(), HD5::Dims::Cartesian);
  }
  Log::Print("Finished {}", parser.GetCommand().Name());
}
