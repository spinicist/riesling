#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "op/nufft.hpp"
#include "parse_args.hpp"
#include "sdc.h"
#include "threads.h"

using namespace rl;

int main_nufft(args::Subparser &parser)
{
  CoreOpts core(parser);
  ExtraOpts extra(parser);
  SDC::Opts sdcOpts(parser);
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::ValueFlag<std::string> trajFile(parser, "T", "Alternative trajectory file for sampling", {"traj"});
  args::ValueFlag<std::string> dset(parser, "D", "Dataset name (channels/noncartesian)", {'d', "dset"});
  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  Trajectory traj;
  if (trajFile) {
    if (!fwd) {
      Log::Fail("Specifying a trajectory file in the adjoint direction is not supported");
    }
    HD5::RieslingReader trajReader(trajFile.Get());
    traj = trajReader.trajectory();
  } else {
    traj = reader.trajectory();
  }
  auto const info = traj.info();
  auto const basis = ReadBasis(core.basisFile);
  auto gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), info.channels, basis);
  NUFFTOp nufft(info.matrix, gridder.get());
  Cx6 channels(AddBack(nufft.inputDimensions(), info.volumes));
  Cx4 noncart(AddBack(nufft.outputDimensions(), info.volumes));

  HD5::Writer writer(OutName(core.iname.Get(), core.oname.Get(), "nufft", "h5"));
  writer.writeTrajectory(traj);

  auto const start = Log::Now();
  if (fwd) {
    std::string const name = dset ? dset.Get() : HD5::Keys::Channels;
    reader.readTensor(name, channels);
    for (auto ii = 0; ii < info.volumes; ii++) {
      noncart.chip<3>(ii).device(Threads::GlobalDevice()) = nufft.forward(channels.chip<5>(ii));
    }
    writer.writeTensor(noncart, HD5::Keys::Noncartesian);
    Log::Print(FMT_STRING("Forward NUFFT took {}"), Log::ToNow(start));
  } else {
    auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
    std::string const name = dset ? dset.Get() : HD5::Keys::Noncartesian;
    reader.readTensor(name, noncart);
    for (auto ii = 0; ii < info.volumes; ii++) {
      channels.chip<5>(ii).device(Threads::GlobalDevice()) = nufft.adjoint(sdc->adjoint(noncart.chip<3>(ii)));
    }
    writer.writeTensor(channels, HD5::Keys::Channels);
    Log::Print(FMT_STRING("NUFFT Adjoint took {}"), Log::ToNow(start));
  }

  return EXIT_SUCCESS;
}
