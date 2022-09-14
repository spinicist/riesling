#include "types.h"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/nufft.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "threads.hpp"

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
  HD5::Writer writer(OutName(core.iname.Get(), core.oname.Get(), "nufft", "h5"));
  writer.writeTrajectory(traj);

  auto const start = Log::Now();
  if (fwd) {
    std::string const name = dset ? dset.Get() : HD5::Keys::Channels;
    auto const channels = reader.readTensor<Cx6>(name);
    auto gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), channels.dimension(0), basis);
    NUFFTOp nufft(info.matrix, gridder.get());
    Cx4 noncart(AddBack(nufft.outputDimensions(), channels.dimension(5)));
    for (auto ii = 0; ii < channels.dimension(5); ii++) {
      noncart.chip<3>(ii).device(Threads::GlobalDevice()) = nufft.forward(channels.chip<5>(ii));
    }
    writer.writeTensor(noncart, HD5::Keys::Noncartesian);
    Log::Print(FMT_STRING("Forward NUFFT took {}"), Log::ToNow(start));
  } else {
    std::string const name = dset ? dset.Get() : HD5::Keys::Noncartesian;
    auto const noncart = reader.readTensor<Cx4>(name);
    auto const channels = noncart.dimension(0);
    auto const sdc = SDC::Choose(sdcOpts, traj, channels, core.ktype.Get(), core.osamp.Get());
    auto gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), channels, basis);
    NUFFTOp nufft(info.matrix, gridder.get());
    Cx6 output(AddBack(nufft.inputDimensions(), noncart.dimension(3)));
    for (auto ii = 0; ii < noncart.dimension(3); ii++) {
      output.chip<5>(ii).device(Threads::GlobalDevice()) = nufft.adjoint(sdc->adjoint(noncart.chip<3>(ii)));
    }
    writer.writeTensor(output, HD5::Keys::Channels);
    Log::Print(FMT_STRING("NUFFT Adjoint took {}"), Log::ToNow(start));
  }

  return EXIT_SUCCESS;
}
