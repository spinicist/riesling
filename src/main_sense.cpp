#include "types.h"

#include "cropper.h"
#include "espirit.h"
#include "fft_plan.h"
#include "filter.h"
#include "io_hd5.h"
#include "io_nifti.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"
#include "sense.h"
#include "tensorOps.h"

int main_sense(args::Subparser &parser)
{
  CORE_RECON_ARGS;

  args::ValueFlag<long> volume(
      parser, "SENSE VOLUME", "Take SENSE maps from this volume (default last)", {"volume"}, -1);
  args::ValueFlag<float> lambda(
      parser, "LAMBDA", "Tikhonov regularisation parameter", {"lambda"}, 0.f);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default header value)", {"fov"}, -1);
  args::ValueFlag<float> res(
      parser, "RESOLUTION", "Resolution for initial gridding (default 8 mm)", {"res", 'r'}, 8.f);
  args::Flag nifti(parser, "NIFTI", "Write output to nifti instead of .h5", {"nii"});

  Log log = ParseCommand(parser, iname);
  FFT::Start(log);

  HD5::Reader reader(iname.Get(), log);
  auto const traj = reader.readTrajectory();
  auto const &info = traj.info();
  Cx3 rad_ks = info.noncartesianVolume();
  reader.readNoncartesian(LastOrVal(volume, info.volumes), rad_ks);
  Cx4 sense = DirectSENSE(traj, osamp.Get(), kb, fov.Get(), rad_ks, lambda.Get(), log);

  auto const fname = OutName(iname.Get(), oname.Get(), "sense", oftype.Get());
  if (oftype.Get().compare("h5") == 0) {
    HD5::Writer writer(fname, log);
    writer.writeInfo(info);
    writer.writeSENSE(sense);
  } else {
    Cx4 const output = FirstToLast4(sense);
    WriteNifti(info, output, fname, log);
  }

  FFT::End(log);
  return EXIT_SUCCESS;
}
