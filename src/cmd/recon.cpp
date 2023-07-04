#include "types.hpp"

#include "cropper.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/recon.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"
#include "tensorOps.hpp"

using namespace rl;

int main_recon(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser, "pipe");
  SENSE::Opts senseOpts(parser);
  args::Flag fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::ValueFlag<std::string> trajName(parser, "T", "Override trajectory", {"traj"});
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj;
  if (trajName) {
    HD5::Reader trajReader(trajName.Get());
    traj = Trajectory(trajReader.readInfo(), trajReader.readTensor<Re3>(HD5::Keys::Trajectory));
  } else {
    traj = Trajectory(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  }
  auto const basis = ReadBasis(coreOpts.basisFile.Get());

  Index volumes = fwd ? reader.dimensions<5>(HD5::Keys::Image)[4] : reader.dimensions<5>(HD5::Keys::Noncartesian)[4];

  if (fwd) {
    HD5::Reader senseReader(senseOpts.type.Get());
    auto const basis = ReadBasis(coreOpts.basisFile.Get());
    auto const sense = std::make_shared<SenseOp>(SENSE::Choose(senseOpts, coreOpts, traj, reader), basis.dimension(0));
    auto const recon = make_recon(coreOpts, sdcOpts, traj, sense, basis);
    Sz4 const sz = recon->ishape;
    Sz4 const osz = AddFront(traj.matrix(coreOpts.fov.Get()), sz[0]);

    auto const &all_start = Log::Now();
    auto const images = reader.readTensor<Cx5>(HD5::Keys::Image);
    Cx4 padded(sz);
    Cx5 kspace(AddBack(recon->oshape, volumes));
    for (Index iv = 0; iv < volumes; iv++) {
      padded.setZero();
      Crop(padded, osz) = images.chip<4>(iv);
      kspace.chip<4>(iv) = recon->forward(padded);
    }
    Log::Print("All Volumes: {}", Log::ToNow(all_start));
    auto const fname = OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "recon", "h5");
    HD5::Writer writer(fname);
    writer.writeInfo(traj.info());
    writer.writeTensor(HD5::Keys::Trajectory, traj.points().dimensions(), traj.points().data());
    writer.writeTensor(HD5::Keys::Noncartesian, kspace.dimensions(), kspace.data());
  } else {
    auto const basis = ReadBasis(coreOpts.basisFile.Get());
    auto const sense = std::make_shared<SenseOp>(SENSE::Choose(senseOpts, coreOpts, traj, reader), basis.dimension(0));
    auto const recon = make_recon(coreOpts, sdcOpts, traj, sense, basis);
    Sz4 const sz = recon->ishape;
    Sz4 const osz = AMin(AddFront(traj.matrix(coreOpts.fov.Get()), sz[0]), sz);
    Cx4 vol(sz);
    Cx5 out(AddBack(osz, volumes));
    out.setZero();
    auto const &all_start = Log::Now();
    for (Index iv = 0; iv < volumes; iv++) {
      vol = recon->adjoint(reader.readSlab<Cx4>(HD5::Keys::Noncartesian, iv));
      out.chip<4>(iv) = Crop(vol, osz);
    }
    Log::Print("All Volumes: {}", Log::ToNow(all_start));
    WriteOutput(coreOpts, out, parser.GetCommand().Name(), traj, Log::Saved());
  }

  return EXIT_SUCCESS;
}
