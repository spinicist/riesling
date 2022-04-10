#include "types.h"

#include "io/io.h"
#include "log.h"
#include "op/grids.h"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"

int main_sense_calib(args::Subparser &parser)
{
  CORE_RECON_ARGS;
  args::ValueFlag<Index> sVol(parser, "V", "SENSE calibration volume", {"senseVolume"}, -1);
  args::ValueFlag<float> sRes(parser, "R", "SENSE calibration res (12 mm)", {"senseRes"}, 12.f);
  args::ValueFlag<float> sReg(parser, "L", "SENSE regularization", {"senseReg"}, 0.f);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default 256 mm)", {"fov"}, 256.f);

  ParseCommand(parser, iname);

  HD5::RieslingReader reader(iname.Get());
  auto const traj = reader.trajectory();
  auto const &info = traj.info();
  auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, fastgrid);
  auto const sdc = SDC::Choose(sdcType.Get(), traj, osamp.Get(), sdcPow.Get());
  Cx4 sense = SelfCalibration(
    info,
    gridder.get(),
    fov.Get(),
    sRes.Get(),
    sReg.Get(),
    sdc->Adj(reader.noncartesian(ValOrLast(sVol.Get(), info.volumes))));

  auto const fname = OutName(iname.Get(), oname.Get(), "sense", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(sense, "sense");

  return EXIT_SUCCESS;
}
