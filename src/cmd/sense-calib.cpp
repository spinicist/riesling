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
  SENSE::Opts            senseOpts(parser);
  SDC::Opts              sdcOpts(parser, "pipe");
  args::ValueFlag<Index> frame(parser, "F", "SENSE calibration frame (all)", {"frame"}, -1);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory  traj(reader.readInfo(), reader.readTensor<Re3>(HD5::Keys::Trajectory));
  auto        noncart = reader.readTensor<Cx5>(HD5::Keys::Noncartesian);
  traj.checkDims(FirstN<3>(noncart.dimensions()));
  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto maps = SENSE::UniformNoise(senseOpts.λ.Get(), SENSE::LoresChannels(senseOpts, coreOpts, traj, noncart, basis));
  if (frame) {
    if (frame.Get() < 0 || frame.Get() >= maps.dimension(1)) {
      Log::Fail("Requested frame {} is outside valid range 0-{}", frame.Get(), maps.dimension(1));
    }
    auto sz = maps.dimensions();
    sz[1] = 1;
    maps = Cx5(maps.slice(Sz5{0, frame.Get(), 0, 0, 0}, sz));
  }
  auto const  fname = OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "sense", "h5");
  HD5::Writer writer(fname);
  writer.writeTensor(HD5::Keys::SENSE, maps.dimensions(), maps.data(), HD5::Dims::SENSE);
  return EXIT_SUCCESS;
}
