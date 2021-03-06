#include "types.h"

#include "apodizer.h"
#include "cropper.h"
#include "espirit.h"
#include "fft_plan.h"
#include "filter.h"
#include "gridder.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "parse_args.h"
#include "sense.h"

int main_espirit(args::Subparser &parser)
{
  CORE_RECON_ARGS;

  args::ValueFlag<long> volume(
      parser, "VOL", "Take SENSE maps from this volume (default last)", {"volume"}, -1);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default header value)", {"fov"}, -1);
  args::ValueFlag<long> kRad(parser, "RAD", "Kernel radius (default 4)", {"kRad", 'k'}, 4);
  args::ValueFlag<long> calRad(
      parser, "RAD", "Additional calibration radius (default 1)", {"calRad", 'c'}, 1);
  args::ValueFlag<float> thresh(
      parser, "T", "Variance threshold to retain kernels (0.015)", {"thresh"}, 0.015);
  args::ValueFlag<float> res(
      parser, "R", "Resolution for initial gridding (default 8 mm)", {"res", 'r'}, 8.f);
  args::Flag nifti(parser, "NIFTI", "Write output to nifti instead of .h5", {"nii"});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);

  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const &info = traj.info();
  Kernel *kernel =
      kb ? (Kernel *)new KaiserBessel(3, osamp.Get(), (info.type == Info::Type::ThreeD))
         : (Kernel *)new NearestNeighbour();

  Cx3 rad_ks = info.noncartesianVolume();
  reader.readNoncartesian(LastOrVal(volume, info.volumes), rad_ks);

  log.info(FMT_STRING("Cropping data to {} mm effective resolution"), res.Get());
  Cx3 lo_ks = rad_ks;
  auto const lo_traj = traj.trim(res.Get(), lo_ks);
  Gridder gridder(lo_traj, osamp.Get(), kernel, false, log);
  SDC::Load("pipe", lo_traj, gridder, log);
  long const totalCalRad =
      kRad.Get() + calRad.Get() + (gridder.info().spokes_lo ? 0 : gridder.info().read_gap);
  Cropper cropper(info, gridder.gridDims(), fov.Get(), log);
  Cx4 sense = cropper.crop4(ESPIRIT(gridder, lo_ks, kRad.Get(), totalCalRad, thresh.Get(), log));

  auto const fname = OutName(iname.Get(), oname.Get(), "espirit", oftype.Get());
  if (oftype.Get().compare("h5") == 0) {
    HD5::Writer writer(fname, log);
    writer.writeInfo(info);
    writer.writeSENSE(sense);
  } else {
    Cx4 const output = SwapToChannelLast(sense);
    WriteNifti(info, output, fname, log);
  }

  FFT::End(log);
  return EXIT_SUCCESS;
}
