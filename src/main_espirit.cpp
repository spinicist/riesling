#include "types.h"

#include "cropper.h"
#include "espirit.h"
#include "filter.h"
#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"

int main_espirit(args::Subparser &parser)
{
  CORE_RECON_ARGS;

  args::ValueFlag<Index> volume(
    parser, "VOL", "Take SENSE maps from this volume (default last)", {"volume"}, -1);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default header value)", {"fov"}, -1);
  args::ValueFlag<Index> readStart(parser, "R", "Reference region start (0)", {"read_start"}, 0);
  args::ValueFlag<Index> kRad(parser, "RAD", "Kernel radius (default 4)", {"kRad", 'k'}, 4);
  args::ValueFlag<Index> calRad(
    parser, "RAD", "Additional calibration radius (default 1)", {"calRad", 'c'}, 1);
  args::ValueFlag<float> thresh(
    parser, "T", "Variance threshold to retain kernels (0.015)", {"thresh"}, 0.015);
  args::ValueFlag<float> res(
    parser, "R", "Resolution for initial gridding (default 8 mm)", {"res", 'r'}, 8.f);

  ParseCommand(parser, iname);
  FFT::Start();

  HD5::RieslingReader reader(iname.Get());
  auto const traj = reader.trajectory();
  auto const &info = traj.info();
  Log::Print(FMT_STRING("Cropping data to {} mm effective resolution"), res.Get());
  auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), osamp.Get(), 0, res, true);
  auto gridder = make_grid(kernel.get(), mapping, fastgrid);
  auto const sdc = SDCPrecond{SDC::Pipe(traj, true, osamp.Get())};
  Index const totalCalRad = kRad.Get() + calRad.Get() + readStart.Get();
  Cropper cropper(info, gridder->mapping().cartDims, fov.Get());
  Cx4 sense = cropper.crop4(ESPIRIT(
    gridder.get(),
    sdc.apply(reader.noncartesian(ValOrLast(volume.Get(), info.volumes))),
    kRad.Get(),
    totalCalRad,
    readStart.Get(),
    thresh.Get()));

  auto const fname = OutName(iname.Get(), oname.Get(), "espirit", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(sense, HD5::Keys::SENSE);

  FFT::End();
  return EXIT_SUCCESS;
}
