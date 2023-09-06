#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/ndft.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "threads.hpp"

using namespace rl;

int main_ndft(args::Subparser &parser)
{
  CoreOpts                     coreOpts(parser);
  SDC::Opts                    sdcOpts(parser, "pipe");
  args::Flag                   fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::ValueFlag<std::string> trajFile(parser, "T", "Alternative trajectory file for sampling", {"traj"});
  args::ValueFlag<std::string> dset(parser, "D", "Dataset name (channels/noncartesian)", {'d', "dset"});
  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());

  auto const  basis = ReadBasis(coreOpts.basisFile.Get());
  HD5::Writer writer(OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "ndft", "h5"));

  auto const start = Log::Now();
  if (fwd) {
    Trajectory traj;
    if (trajFile) {
      HD5::Reader trajReader(trajFile.Get());
      traj = Trajectory(trajReader.readInfo(), trajReader.readTensor<Re3>(HD5::Keys::Trajectory));
    } else {
      traj = Trajectory(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
    }
    std::string const name = dset ? dset.Get() : HD5::Keys::Channels;
    auto              channels = reader.readTensor<Cx6>(name);
    auto              ndft = make_ndft(traj.points(), channels.dimension(0), traj.matrixForFOV(coreOpts.fov.Get()), basis);
    Cx5               noncart(AddBack(ndft->oshape, channels.dimension(5)));
    for (auto ii = 0; ii < channels.dimension(5); ii++) {
      noncart.chip<4>(ii).chip<3>(0).device(Threads::GlobalDevice()) = ndft->forward(CChipMap(channels, ii));
    }
    writer.writeTensor(HD5::Keys::Noncartesian, noncart.dimensions(), noncart.data());
    writer.writeInfo(traj.info());
    writer.writeTensor(HD5::Keys::Trajectory, traj.points().dimensions(), traj.points().data());
    Log::Print("Forward NDFT took {}", Log::ToNow(start));
  } else {
    Trajectory        traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
    std::string const name = dset ? dset.Get() : HD5::Keys::Noncartesian;
    auto              noncart = reader.readTensor<Cx5>(name);
    traj.checkDims(FirstN<3>(noncart.dimensions()));
    auto const channels = noncart.dimension(0);
    auto const sdc = SDC::Choose(sdcOpts, channels, traj, coreOpts.ktype.Get(), coreOpts.osamp.Get());
    auto       ndft = make_ndft(traj.points(), channels, traj.matrixForFOV(coreOpts.fov.Get()), basis, sdc);
    Cx6 output(AddBack(ndft->ishape, noncart.dimension(3)));
    for (auto ii = 0; ii < noncart.dimension(4); ii++) {
      output.chip<5>(ii).device(Threads::GlobalDevice()) = ndft->adjoint(CChipMap(noncart, ii));
    }
    writer.writeTensor(HD5::Keys::Channels, output.dimensions(), output.data());
    Log::Print("NDFT Adjoint took {}", Log::ToNow(start));
  }

  return EXIT_SUCCESS;
}
