#include "types.hpp"

#include "io/hd5.hpp"
#include "log.hpp"
#include "op/gridBase.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense.h"

using namespace rl;

int main_sense_calib(args::Subparser &parser)
{
  CoreOpts core(parser);
  SDC::Opts sdcOpts(parser);
  args::ValueFlag<Index> volume(parser, "V", "SENSE calibration volume", {"sense-vol"}, 0);
  args::ValueFlag<Index> frame(parser, "F", "SENSE calibration frame", {"sense-frame"}, 0);
  args::ValueFlag<float> res(parser, "R", "SENSE calibration res (12 mm)", {"sense-res"}, 12.f);
  args::ValueFlag<float> λ(parser, "L", "SENSE regularization", {"sense-lambda"}, 0.f);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default 256 mm)", {"fov"}, 256.f);

  ParseCommand(parser, core.iname);

  HD5::Reader reader(core.iname.Get());
  Trajectory traj(reader);
  Cx3 const data = reader.readSlab<Cx3>(HD5::Keys::Noncartesian, volume.Get());
  Index const channels = data.dimension(0);
  auto const basis = ReadBasis(core.basisFile);
  auto gridder = make_grid<Cx, 3>(traj, core.ktype.Get(), core.osamp.Get(), channels, basis);
  auto const sdc = SDC::Choose(sdcOpts, traj, channels, core.ktype.Get(), core.osamp.Get());
  Cx4 sense = SENSE::SelfCalibration(traj, gridder.get(), fov.Get(), res.Get(), λ.Get(), frame.Get(), sdc->adjoint(data));
  auto const fname = OutName(core.iname.Get(), core.oname.Get(), "sense", "h5");
  HD5::Writer writer(fname);
  writer.writeTensor(sense, "sense");

  return EXIT_SUCCESS;
}
