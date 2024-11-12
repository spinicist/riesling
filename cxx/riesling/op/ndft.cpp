#include "types.hpp"

#include "algo/lsmr.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grid.hpp"
#include "op/ndft.hpp"
#include "precon.hpp"
#include "sys/threads.hpp"

using namespace rl;

void main_ndft(args::Subparser &parser)
{
  CoreArgs    coreArgs(parser);
  GridArgs<3> gridArgs(parser);
  PreconArgs  preArgs(parser);
  LsqOpts     lsqOpts(parser);

  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  ParseCommand(parser, coreArgs.iname, coreArgs.oname);

  HD5::Reader reader(coreArgs.iname.Get());
  Info const  info = reader.readInfo();
  auto const  basis = LoadBasis(coreArgs.basisFile.Get());
  HD5::Writer writer(coreArgs.oname.Get());
  writer.writeInfo(info);

  Trajectory traj(reader, info.voxel_size, coreArgs.matrix.Get());
  auto const nC = reader.dimensions()[0];
  auto const ndft = TOps::NDFT<3>::Make(traj.matrixForFOV(gridArgs.fov.Get()), traj.points(), nC, basis.get());

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
    Index const nS = noncart.dimension(3);
    Index const nT = noncart.dimension(4);
    traj.checkDims(FirstN<3>(noncart.dimensions()));

    auto const M = MakeKspacePre(preArgs.Get(), gridArgs.Get(), traj, nC, nS, nT, basis.get());
    LSMR const lsmr{ndft, M, nullptr, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};

    Cx6 output(AddBack(ndft->ishape, noncart.dimension(3)));
    for (auto ii = 0; ii < noncart.dimension(4); ii++) {
      output.chip<5>(ii).device(Threads::TensorDevice()) = AsTensorMap(lsmr.run(CollapseToArray(noncart)), ndft->ishape);
    }
    writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), HD5::Dims::Channels);
  }
}
