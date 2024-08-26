#include "types.hpp"

#include "algo/lsmr.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grid.hpp"
#include "op/ndft.hpp"
#include "precon.hpp"
#include "threads.hpp"

using namespace rl;

void main_ndft(args::Subparser &parser)
{
  CoreOpts   coreOpts(parser);
  GridOpts   gridOpts(parser);
  PreconOpts preOpts(parser);
  LsqOpts    lsqOpts(parser);

  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  auto const basis = LoadBasis(coreOpts.basisFile.Get());
  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeInfo(info);

  Trajectory traj(reader, info.voxel_size);
  auto const nC = reader.dimensions()[0];
  auto const ndft = TOps::NDFT<3>::Make(traj.matrixForFOV(coreOpts.fov.Get()), traj.points(), nC, basis.get());

  if (fwd) {
    auto channels = reader.readTensor<Cx6>();
    Cx5  noncart(AddBack(ndft->oshape, 1, channels.dimension(5)));
    for (auto ii = 0; ii < channels.dimension(5); ii++) {
      noncart.chip<4>(ii).chip<3>(0).device(Threads::TensorDevice()) = ndft->forward(CChipMap(channels, ii));
    }
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
    traj.write(writer);
  } else {
    auto        noncart = reader.readTensor<Cx5>();
    Index const nT = noncart.dimension(4);
    traj.checkDims(FirstN<3>(noncart.dimensions()));

    auto const M = MakeKspacePre(traj, nC, nT, basis.get(), preOpts.type.Get(), preOpts.bias.Get());
    LSMR const lsmr{ndft, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};

    Cx6 output(AddBack(ndft->ishape, noncart.dimension(3)));
    for (auto ii = 0; ii < noncart.dimension(4); ii++) {
      output.chip<5>(ii).device(Threads::TensorDevice()) = Tensorfy(lsmr.run(CollapseToArray(noncart)), ndft->ishape);
    }
    writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), HD5::Dims::Channels);
  }
}
