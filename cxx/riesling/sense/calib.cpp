#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"

using namespace rl;

void main_sense_calib(args::Subparser &parser)
{
  CoreOpts                     coreOpts(parser);
  GridOpts                     gridOpts(parser);
  SENSE::Opts                  senseOpts(parser);
  args::ValueFlag<std::string> refname(parser, "F", "Reference scan filename", {"ref"});
  args::ValueFlag<Index>       frame(parser, "F", "SENSE calibration frame (all)", {"frame"}, -1);

  ParseCommand(parser, coreOpts.iname, coreOpts.oname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  auto const basis = ReadBasis(coreOpts.basisFile.Get());

  Cx5 channels = SENSE::LoresChannels(senseOpts, gridOpts, traj, noncart, basis);

  if (refname) {
    HD5::Reader refFile(refname.Get());
    Trajectory  refTraj(refFile, refFile.readInfo().voxel_size);
    if (!refTraj.compatible(traj)) { Log::Fail("Reference data incompatible with multi-channel data"); }
    auto refNoncart = refFile.readTensor<Cx5>();
    if (refNoncart.dimension(0) != 1) { Log::Fail("Reference data must be single channel"); }
    refTraj.checkDims(FirstN<3>(refNoncart.dimensions()));
    Cx4 const ref = SENSE::LoresChannels(senseOpts, gridOpts, refTraj, refNoncart, basis).chip<0>(0);
    SENSE::RegularizedNormalization(senseOpts.λ.Get(), ref, channels);
  }

  SENSE::RegularizedNormalization(senseOpts.λ.Get(), channels);
  if (frame) {
    auto shape = channels.dimensions();
    if (frame.Get() < 0 || frame.Get() >= shape[1]) {
      Log::Fail("Requested frame {} is outside valid range 0-{}", frame.Get(), shape[1]);
    }
    shape[1] = 1;
    channels = Cx5(channels.slice(Sz5{0, frame.Get(), 0, 0, 0}, shape));
  }
  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeTensor(HD5::Keys::Data, channels.dimensions(), channels.data(), HD5::Dims::SENSE);
  Log::Print("Finished {}", parser.GetCommand().Name());
}
