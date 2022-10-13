#include "types.hpp"

#include "cropper.h"
#include "espirit.h"
#include "io/hd5.hpp"
#include "log.hpp"
#include "mapping.hpp"
#include "op/make_grid.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "sense.hpp"
#include "tensorOps.hpp"

using namespace rl;

int main_espirit(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser);
  args::ValueFlag<Index> volume(parser, "VOL", "Take SENSE maps from this volume (default first)", {"sense-vol"}, 0);
  args::ValueFlag<float> res(parser, "R", "Resolution for initial gridding (default 12 mm)", {"sense-res", 'r'}, 12.f);
  args::ValueFlag<float> fov(parser, "FOV", "FoV in mm (default header value)", {"fov"}, -1);
  args::ValueFlag<Index> lores(parser, "L", "Lo-res traces", {"lores"}, 0);
  args::ValueFlag<Index> readStart(parser, "R", "Reference region start (0)", {"read-start"}, 0);
  args::ValueFlag<Index> kRad(parser, "RAD", "Kernel radius (default 4)", {"krad", 'k'}, 4);
  args::ValueFlag<Index> calRad(parser, "RAD", "Additional calibration radius (default 1)", {"calRad", 'c'}, 1);
  args::ValueFlag<float> thresh(parser, "T", "Variance threshold to retain kernels (0.015)", {"thresh"}, 0.015);

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader);
  auto const ks1 = reader.readSlab<Cx4>(HD5::Keys::Noncartesian, volume.Get());
  auto const &info = traj.info();
  Log::Print(FMT_STRING("Cropping data to {} mm effective resolution"), res.Get());
  auto const [dsTraj, ks] = traj.downsample(ks1, res.Get(), lores.Get(), false);
  auto const dsInfo = dsTraj.info();
  auto gridder = make_grid<Cx, 3>(dsTraj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), ks.dimension(0));
  auto const sdc = SDC::make_sdc(sdcOpts, dsTraj, ks1.dimension(0), coreOpts.ktype.Get(), coreOpts.osamp.Get());
  Index const totalCalRad = kRad.Get() + calRad.Get() + readStart.Get();
  Cropper cropper(traj.info().matrix, LastN<3>(gridder->inputDimensions()), traj.info().voxel_size, fov.Get());
  Cx4 sense =
    cropper.crop4(ESPIRIT(gridder.get(), (*sdc)(ks.chip<3>(0)), kRad.Get(), totalCalRad, readStart.Get(), thresh.Get()));

  auto const fname = OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "espirit", "h5");
  HD5::Writer writer(fname);
  writer.writeInfo(info);
  writer.writeTensor(sense, HD5::Keys::SENSE);

  return EXIT_SUCCESS;
}
