#include "types.h"

#include "io.h"
#include "log.h"
#include "op/nufft.hpp"
#include "parse_args.h"
#include "threads.h"

int main_nufft(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  args::Flag forward(parser, "F", "Apply forward (default reverse)", {'f', "fwd"});
  args::ValueFlag<std::string> dset(
    parser, "D", "Dataset name (image/noncartesian)", {'d', "dset"});
  Log log = ParseCommand(parser, iname);
  FFT::Start(log);
  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const info = traj.info();

  auto gridder = make_grid(traj, osamp.Get(), kernel.Get(), fastgrid, log);
  if (!forward) {
    gridder->setSDC(SDC::Choose(sdc.Get(), traj, osamp.Get(), log));
    gridder->setSDCPower(sdcPow.Get());
  }

  NufftOp nufft(gridder.get(), info.channels, 1, info.matrix, log);
  Cx5 channels{nufft.inputDimensions()};
  Cx3 noncart{nufft.outputDimensions()};
  HD5::Writer writer(OutName(iname.Get(), oname.Get(), "nufft", "h5"), log);
  writer.writeTrajectory(traj);
  auto const start = log.now();
  if (forward) {
    std::string const name = dset ? dset.Get() : "image";
    auto const input = reader.readTensor<Cx4>(name);
    channels = input.reshape(
      Sz5{input.dimension(0), 1, input.dimension(1), input.dimension(2), input.dimension(3)});
    nufft.A(channels, noncart);
    writer.writeTensor(noncart, "nufft-forward");
    log.info("Forward NUFFT took {}", log.toNow(start));
  } else {
    std::string const name = dset ? dset.Get() : "nufft-forward";
    noncart = reader.readTensor<Cx3>(name);
    nufft.Adj(noncart, channels);
    Cx4 output = channels.reshape(Sz4{
      channels.dimension(0), channels.dimension(2), channels.dimension(3), channels.dimension(4)});
    writer.writeTensor(output, "nufft-backward");
    log.info("Backward NUFFT took {}", log.toNow(start));
  }

  FFT::End(log);
  return EXIT_SUCCESS;
}
