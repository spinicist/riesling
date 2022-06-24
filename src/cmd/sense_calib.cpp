#include "types.h"

#include "io/hd5.hpp"
#include "log.h"
#include "op/grids.h"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"

int main_sense_calib(args::Subparser &parser)
{
  CoreOpts core(parser);
  SDC::Opts sdcOpts(parser);
  args::ValueFlag<Index> volume(parser, "V", "SENSE calibration volume", {"sense-vol"}, -1);
  args::ValueFlag<Index> frame(parser, "F", "SENSE calibration frame", {"sense-frame"}, 0);
  args::ValueFlag<float> res(parser, "R", "SENSE calibration res (12 mm)", {"sense-res"}, 12.f);
  args::ValueFlag<float> λ(parser, "L", "SENSE regularization", {"sense-lambda"}, 0.f);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default 256 mm)", {"fov"}, 256.f);

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  auto const traj = reader.trajectory();
  auto const &info = traj.info();
  auto const kernel = make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), core.osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, info.channels, core.fast);
  auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
  Cx3 const data = sdc->Adj(reader.noncartesian(ValOrLast(volume.Get(), info.volumes)));
  Cx4 sense = SENSE::SelfCalibration(info, gridder.get(), fov.Get(), res.Get(), λ.Get(), frame.Get(), data);

  auto const fname = OutName(core.iname.Get(), core.oname.Get(), "sense", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(sense, "sense");

  return EXIT_SUCCESS;
}
