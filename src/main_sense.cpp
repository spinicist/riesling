#include "types.h"

#include "cropper.h"
#include "espirit.h"
#include "fft_plan.h"
#include "filter.h"
#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"
#include "sense.h"
#include "tensorOps.h"

int main_sense(args::Subparser &parser)
{
  CORE_RECON_ARGS;

  args::ValueFlag<long> vol(
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
  auto gridder = make_grid(traj, osamp.Get(), kernel.Get(), fastgrid, log);
  gridder->setSDC(SDC::Choose(sdc.Get(), traj, osamp.Get(), log));
  Cx4 sense = DirectSENSE(
    info,
    gridder.get(),
    fov.Get(),
    lambda.Get(),
    reader.noncartesian(ValOrLast(vol.Get(), info.volumes)),
    log);

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
