#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/grid.hpp"
#include "op/ndft.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "threads.hpp"

using namespace rl;

int main_ndft(args::Subparser &parser)
{
  CoreOpts   coreOpts(parser);
  GridOpts   gridOpts(parser);
  SDC::Opts  sdcOpts(parser, "pipe");
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const info = reader.readInfo();
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeInfo(info);
  auto const        start = Log::Now();
  Trajectory traj(reader, info.voxel_size);
  if (fwd) {
    auto channels = reader.readTensor<Cx6>();
    auto ndft = make_ndft(traj.points(), channels.dimension(0), traj.matrixForFOV(coreOpts.fov.Get()), basis);
    Cx5  noncart(AddBack(ndft->oshape, channels.dimension(5)));
    for (auto ii = 0; ii < channels.dimension(5); ii++) {
      noncart.chip<4>(ii).chip<3>(0).device(Threads::GlobalDevice()) = ndft->forward(CChipMap(channels, ii));
    }
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
    traj.write(writer);
    Log::Print("Forward NDFT took {}", Log::ToNow(start));
  } else {
    auto noncart = reader.readTensor<Cx5>();
    traj.checkDims(FirstN<3>(noncart.dimensions()));
    auto const channels = noncart.dimension(0);
    auto const sdc = SDC::Choose(sdcOpts, channels, traj, gridOpts.ktype.Get(), gridOpts.osamp.Get());
    auto       ndft = make_ndft(traj.points(), channels, traj.matrixForFOV(coreOpts.fov.Get()), basis, sdc);
    Cx6        output(AddBack(ndft->ishape, noncart.dimension(3)));
    for (auto ii = 0; ii < noncart.dimension(4); ii++) {
      output.chip<5>(ii).device(Threads::GlobalDevice()) = ndft->adjoint(CChipMap(noncart, ii));
    }
    writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data());
    Log::Print("NDFT Adjoint took {}", Log::ToNow(start));
  }

  return EXIT_SUCCESS;
}
