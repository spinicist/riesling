#include "types.hpp"

#include "algo/lsmr.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/nufft.hpp"
#include "precon.hpp"
#include "sys/threads.hpp"

using namespace rl;

void main_nufft(args::Subparser &parser)
{
  CoreOpts   coreOpts(parser);
  GridOpts   gridOpts(parser);
  PreconOpts preOpts(parser);
  LsqOpts    lsqOpts(parser);

  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);
  auto const  cmd = parser.GetCommand().Name();
  HD5::Reader reader(coreOpts.iname.Get());

  Trajectory traj(reader, reader.readInfo().voxel_size);
  auto const basis = LoadBasis(coreOpts.basisFile.Get());

  auto const nC = reader.dimensions()[1];
  auto const nufft = TOps::NUFFT<3>::Make(traj, gridOpts, nC, basis.get(), traj.matrixForFOV(coreOpts.fov.Get()));

  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeInfo(reader.readInfo());
  traj.write(writer);

  if (fwd) {
    auto const channels = reader.readTensor<Cx6>();
    Cx5        noncart(AddBack(nufft->oshape, 1, channels.dimension(5)));
    for (auto ii = 0; ii < channels.dimension(5); ii++) {
      auto imap = CChipMap(channels, ii);
      auto omapt = ChipMap(noncart, ii);
      auto omaps = ChipMap(omapt, 0);
      omaps = nufft->forward(imap);
    }
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
  } else {
    auto const noncart = reader.readTensor<Cx5>();
    traj.checkDims(FirstN<3>(noncart.dimensions()));

    auto const M = MakeKspacePre(traj, nC, 1, basis.get(), preOpts.type.Get(), preOpts.bias.Get());
    LSMR const lsmr{nufft, M, lsqOpts.its.Get(), lsqOpts.atol.Get(), lsqOpts.btol.Get(), lsqOpts.ctol.Get()};

    Cx6 output(AddBack(nufft->ishape, noncart.dimension(3)));
    for (auto ii = 0; ii < noncart.dimension(4); ii++) {
      output.chip<5>(ii).device(Threads::TensorDevice()) = Tensorfy(lsmr.run(CollapseToConstVector(noncart)), nufft->ishape);
    }
    writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), HD5::Dims::Channels);
  }
  Log::Print(cmd, "Finished");
}
