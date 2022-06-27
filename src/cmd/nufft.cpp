#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "op/nufft.hpp"
#include "parse_args.h"
#include "sdc.h"
#include "threads.h"

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
  auto const kernel = make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  Mapping const mapping(traj, kernel.get(), core.osamp.Get(), core.bucketSize.Get());
  auto gridder = make_grid(kernel.get(), mapping, info.channels, core.basisFile.Get());
  NUFFTOp nufft(LastN<3>(gridder->inputDimensions()), gridder.get());
  Cx6 channels(AddBack(nufft.inputDimensions(), info.volumes));
  Cx4 noncart(AddBack(nufft.outputDimensions(), info.volumes));

  HD5::Writer writer(OutName(core.iname.Get(), core.oname.Get(), "nufft", "h5"));
  writer.writeTrajectory(traj);

  auto const start = Log::Now();
  if (fwd) {
    std::string const name = dset ? dset.Get() : HD5::Keys::Channels;
    reader.readTensor(name, channels);
    for (auto ii = 0; ii < info.volumes; ii++) {
      noncart.chip<3>(ii).device(Threads::GlobalDevice()) = nufft.A(channels.chip<5>(ii));
    }
    writer.writeTensor(noncart, HD5::Keys::Noncartesian);
    Log::Print(FMT_STRING("Forward NUFFT took {}"), Log::ToNow(start));
  } else {
    auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
    std::string const name = dset ? dset.Get() : HD5::Keys::Noncartesian;
    reader.readTensor(name, noncart);
    for (auto ii = 0; ii < info.volumes; ii++) {
      channels.chip<5>(ii).device(Threads::GlobalDevice()) = nufft.Adj(sdc->Adj(noncart.chip<3>(ii)));
    }
    writer.writeTensor(channels, HD5::Keys::Channels);
    Log::Print(FMT_STRING("NUFFT Adjoint took {}"), Log::ToNow(start));
  }

  return EXIT_SUCCESS;
}
