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

int main_recon_sense(args::Subparser &parser)
{
  CoreOpts                     coreOpts(parser);
  GridOpts                     gridOpts(parser);
  SDC::Opts                    sdcOpts(parser, "pipe");
  SENSE::Opts                  senseOpts(parser);
  args::Flag                   fwd(parser, "", "Apply forward operation", {'f', "fwd"});
  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Info const  info = reader.readInfo();
  Trajectory  traj(reader, info.voxel_size);
  auto const  basis = ReadBasis(coreOpts.basisFile.Get());

  Index volumes = fwd ? reader.dimensions().at(4) : reader.dimensions().at(4);

  if (fwd) {
    HD5::Reader senseReader(senseOpts.type.Get());
    auto const  sense = std::make_shared<SenseOp>(SENSE::Choose(senseOpts, gridOpts, traj, Cx5()), basis.dimension(0));
    auto const  recon = make_recon(coreOpts, gridOpts, sdcOpts, traj, sense, basis);
    Sz4 const   sz = recon->ishape;
    Sz4 const   osz = AddFront(traj.matrixForFOV(coreOpts.fov.Get()), sz[0]);

    auto const &all_start = Log::Now();
    auto const  images = reader.readTensor<Cx5>();
    Cx4         padded(sz);
    Cx5         kspace(AddBack(recon->oshape, volumes));
    for (Index iv = 0; iv < volumes; iv++) {
      padded.setZero();
      Crop(padded, osz) = images.chip<4>(iv);
      kspace.chip<4>(iv) = recon->forward(padded);
    }
    Log::Print("All Volumes: {}", Log::ToNow(all_start));
    HD5::Writer writer(coreOpts.oname.Get());
    writer.writeInfo(info);
    traj.write(writer);
    writer.writeTensor(HD5::Keys::Data, kspace.dimensions(), kspace.data(), HD5::Dims::Noncartesian);
  } else {
    auto noncart = reader.readTensor<Cx5>();
    traj.checkDims(FirstN<3>(noncart.dimensions()));
    auto const sense = std::make_shared<SenseOp>(SENSE::Choose(senseOpts, gridOpts, traj, noncart), basis.dimension(0));
    auto const recon = make_recon(coreOpts, gridOpts, sdcOpts, traj, sense, basis);
    Sz4 const  sz = recon->ishape;
    Sz4 const  osz = AMin(AddFront(traj.matrixForFOV(coreOpts.fov.Get()), sz[0]), sz);
    Cx4        vol(sz);
    Cx5        out(AddBack(osz, volumes));
    out.setZero();
    for (Index iv = 0; iv < volumes; iv++) {
      vol = recon->adjoint(noncart.chip<4>(iv));
      out.chip<4>(iv) = Crop(vol, osz);
    }
    WriteOutput(coreOpts, out, info, Log::Saved());
  }

  return EXIT_SUCCESS;
}
