#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense/sense.hpp"

using namespace rl;

int main_sense_calib(args::Subparser &parser)
{
  CoreOpts               coreOpts(parser);
  GridOpts               gridOpts(parser);
  SENSE::Opts            senseOpts(parser);
  SDC::Opts              sdcOpts(parser, "pipe");
  args::ValueFlag<Index> frame(parser, "F", "SENSE calibration frame (all)", {"frame"}, -1);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader, reader.readInfo().voxel_size);
  auto        noncart = reader.readTensor<Cx5>();
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto       maps = SENSE::UniformNoise(senseOpts.λ.Get(), SENSE::LoresChannels(senseOpts, gridOpts, traj, noncart, basis));
  if (frame) {
    if (frame.Get() < 0 || frame.Get() >= maps.dimension(1)) {
      Log::Fail("Requested frame {} is outside valid range 0-{}", frame.Get(), maps.dimension(1));
    }
    auto sz = maps.dimensions();
    sz[1] = 1;
    maps = Cx5(maps.slice(Sz5{0, frame.Get(), 0, 0, 0}, sz));
  }
  HD5::Writer writer(coreOpts.oname.Get());
  writer.writeTensor(HD5::Keys::Data, maps.dimensions(), maps.data(), HD5::Dims::SENSE);
  return EXIT_SUCCESS;
}
