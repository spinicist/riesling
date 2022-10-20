#include "types.hpp"

#include "cropper.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/nufft.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "tensorOps.hpp"

using namespace rl;

int main_rss(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser);
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  Index const nC = reader.dimensions<5>(HD5::Keys::Noncartesian)[0];
  auto const sdc = SDC::Choose(sdcOpts, traj, nC, coreOpts.ktype.Get(), coreOpts.osamp.Get());
  auto nufft =
    make_nufft(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), nC, traj.matrix(coreOpts.fov.Get()), basis, sdc, false);
  Sz4 sz = LastN<4>(nufft->inputDimensions());

  Cx5 allData = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  Index const volumes = allData.dimension(4);
  Cx5 out(AddBack(sz, volumes));
  auto const &all_start = Log::Now();
  for (Index iv = 0; iv < volumes; iv++) {
    auto const channels = nufft->adjoint(CChipMap(allData, iv));
    out.chip<4>(iv) = ConjugateSum(channels, channels).sqrt();
  }
  Log::Print(FMT_STRING("All Volumes: {}"), Log::ToNow(all_start));
  WriteOutput(out, coreOpts.iname.Get(), coreOpts.oname.Get(), parser.GetCommand().Name(), coreOpts.keepTrajectory, traj);

  return EXIT_SUCCESS;
}
