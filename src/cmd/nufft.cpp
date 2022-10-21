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
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser);
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::ValueFlag<std::string> trajFile(parser, "T", "Alternative trajectory file for sampling", {"traj"});
  args::ValueFlag<std::string> dset(parser, "D", "Dataset name (channels/noncartesian)", {'d', "dset"});
  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj;
  if (trajFile) {
    if (!fwd) {
      Log::Fail("Specifying a trajectory file in the adjoint direction is not supported");
    }
    HD5::Reader trajReader(trajFile.Get());
    traj = Trajectory(trajReader);
  } else {
    traj = Trajectory(reader);
  }
  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "nufft", "h5"));
  traj.write(writer);

  auto const start = Log::Now();
  if (fwd) {
    std::string const name = dset ? dset.Get() : HD5::Keys::Channels;
    auto channels = reader.readTensor<Cx6>(name);
    auto nufft = make_nufft(
      traj,
      coreOpts.ktype.Get(),
      coreOpts.osamp.Get(),
      channels.dimension(0),
      traj.matrix(coreOpts.fov.Get()),
      basis,
      std::make_shared<IdentityFunctor<Cx3>>(),
      false);
    Cx5 noncart(AddBack(nufft->outputDimensions(), channels.dimension(5)));
    for (auto ii = 0; ii < channels.dimension(5); ii++) {
      noncart.chip<4>(ii).chip<3>(0).device(Threads::GlobalDevice()) = nufft->forward(CChipMap(channels, ii));
    }
    writer.writeTensor(noncart, HD5::Keys::Noncartesian);
    Log::Print(FMT_STRING("Forward NUFFT took {}"), Log::ToNow(start));
  } else {
    std::string const name = dset ? dset.Get() : HD5::Keys::Noncartesian;
    auto noncart = reader.readTensor<Cx5>(name);
    auto const channels = noncart.dimension(0);
    auto const sdc = SDC::Choose(sdcOpts, traj, channels, coreOpts.ktype.Get(), coreOpts.osamp.Get());
    auto nufft = make_nufft(
      traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), channels, traj.matrix(coreOpts.fov.Get()), basis, sdc, false);

    Cx6 output(AddBack(nufft->inputDimensions(), noncart.dimension(3)));
    for (auto ii = 0; ii < noncart.dimension(4); ii++) {
      output.chip<5>(ii).device(Threads::GlobalDevice()) = nufft->adjoint(CChipMap(noncart, ii));
    }
    writer.writeTensor(output, HD5::Keys::Channels);
    Log::Print(FMT_STRING("NUFFT Adjoint took {}"), Log::ToNow(start));
  }

  return EXIT_SUCCESS;
}
