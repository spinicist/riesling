#include "types.h"

#include "io/io.h"
#include "log.h"
#include "op/grids.h"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"

int main_sense_calib(args::Subparser &parser)
{
  CoreOpts core(parser);
  SDC::Opts sdcOpts(parser);
  args::ValueFlag<Index> sVol(parser, "V", "SENSE calibration volume", {"senseVolume"}, -1);
  args::ValueFlag<float> sRes(parser, "R", "SENSE calibration res (12 mm)", {"senseRes"}, 12.f);
  args::ValueFlag<float> sReg(parser, "L", "SENSE regularization", {"senseReg"}, 0.f);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default 256 mm)", {"fov"}, 256.f);

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  auto const traj = reader.trajectory();
  auto const &info = traj.info();
  auto const kernel = make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), core.osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, core.fast);
  auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
  Cx3 const data = reader.noncartesian(ValOrLast(sVol.Get(), info.volumes));
  Cx4 sense =
    SENSE::SelfCalibration(info, gridder.get(), fov.Get(), sRes.Get(), sReg.Get(), sdc ? sdc->Adj(data) : data);

  auto const fname = OutName(core.iname.Get(), core.oname.Get(), "sense", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(sense, "sense");

  return EXIT_SUCCESS;
}
