#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/nufft.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "threads.hpp"

using namespace rl;

int main_nufft(args::Subparser &parser)
{
  CoreOpts   coreOpts(parser);
  GridOpts   gridOpts(parser);
  SDC::Opts  sdcOpts(parser, "pipe");
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());

  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  HD5::Writer writer(coreOpts.oname.Get());

  auto const start = Log::Now();
  Trajectory traj(reader, reader.readInfo().voxel_size);
  if (fwd) {
    auto channels = reader.readTensor<Cx6>();
    auto nufft = make_nufft(traj, gridOpts, channels.dimension(0), traj.matrixForFOV(coreOpts.fov.Get()), basis);
    Cx5  noncart(AddBack(nufft->oshape, channels.dimension(5)));
    for (auto ii = 0; ii < channels.dimension(5); ii++) {
      noncart.chip<4>(ii).chip<3>(0).device(Threads::GlobalDevice()) = nufft->forward(CChipMap(channels, ii));
    }
    writer.writeTensor(HD5::Keys::Data, noncart.dimensions(), noncart.data(), HD5::Dims::Noncartesian);
    writer.writeInfo(reader.readInfo());
    traj.write(writer);
    Log::Print("Forward NUFFT took {}", Log::ToNow(start));
  } else {
    auto noncart = reader.readTensor<Cx5>();
    traj.checkDims(FirstN<3>(noncart.dimensions()));
    auto const channels = noncart.dimension(0);
    auto const sdc = SDC::Choose(sdcOpts, channels, traj, gridOpts.ktype.Get(), gridOpts.osamp.Get());
    auto       nufft = make_nufft(traj, gridOpts, channels, traj.matrixForFOV(coreOpts.fov.Get()), basis, sdc);

    Cx6 output(AddBack(nufft->ishape, noncart.dimension(3)));
    for (auto ii = 0; ii < noncart.dimension(4); ii++) {
      output.chip<5>(ii).device(Threads::GlobalDevice()) = nufft->adjoint(CChipMap(noncart, ii));
    }
    writer.writeTensor(HD5::Keys::Data, output.dimensions(), output.data(), HD5::Dims::Cartesian);
    Log::Print("NUFFT Adjoint took {}", Log::ToNow(start));
  }

  return EXIT_SUCCESS;
}
