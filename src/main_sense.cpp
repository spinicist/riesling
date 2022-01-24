#include "types.h"

#include "cropper.h"
#include "espirit.h"
#include "fft_plan.h"
#include "filter.h"
#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"
#include "sdc.h"
#include "sense.h"
#include "tensorOps.h"

int main_sense(args::Subparser &parser)
{
  CORE_RECON_ARGS;

  args::ValueFlag<Index> vol(
    parser, "SENSE VOLUME", "Take SENSE maps from this volume (default last)", {"volume"}, -1);
  args::ValueFlag<float> lambda(
    parser, "LAMBDA", "Tikhonov regularisation parameter", {"lambda"}, 0.f);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default header value)", {"fov"}, -1);
  args::ValueFlag<float> res(
    parser, "RESOLUTION", "Resolution for initial gridding (default 8 mm)", {"res", 'r'}, 8.f);
  args::Flag nifti(parser, "NIFTI", "Write output to nifti instead of .h5", {"nii"});

  ParseCommand(parser, iname);
  FFT::Start();

  HD5::RieslingReader reader(iname.Get());
  auto const traj = reader.trajectory();
  auto const &info = traj.info();
  auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), osamp.Get());
  auto gridder = make_grid(kernel.get(), mapping, fastgrid);
  gridder->setSDC(SDC::Choose(sdc.Get(), traj, osamp.Get()));
  Cx4 sense = DirectSENSE(
    info,
    gridder.get(),
    fov.Get(),
    lambda.Get(),
    reader.noncartesian(ValOrLast(vol.Get(), info.volumes)));

  auto const fname = OutName(iname.Get(), oname.Get(), "sense", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(sense, "sense");
  FFT::End();
  return EXIT_SUCCESS;
}
