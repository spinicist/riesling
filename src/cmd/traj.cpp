#include "types.h"

#include "fft/fft.hpp"
#include "io/hd5.hpp"
#include "log.h"
#include "op/grids.h"
#include "parse_args.h"
#include "sdc.h"
#include "tensorOps.h"
#include "threads.h"
#include <complex>

int main_traj(args::Subparser &parser)
{
  CoreOpts core(parser);
  SDC::Opts sdcOpts(parser);

  args::ValueFlag<std::string> basisFile(parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});
  args::Flag savePSF(parser, "PSF", "Write out Point-Spread-Function", {"psf", 'p'});

  ParseCommand(parser, core.iname);

  HD5::RieslingReader reader(core.iname.Get());
  auto const inTraj = reader.trajectory();
  // Ensure only one channel for sanity
  auto info = inTraj.info();
  info.channels = 1;
  Trajectory traj(info, inTraj.points(), inTraj.frames());

  auto const kernel = make_kernel(core.ktype.Get(), info.type, core.osamp.Get());
  auto const mapping = traj.mapping(kernel->inPlane(), core.osamp.Get());
  Cx3 rad_ks(1, info.read_points, info.spokes);
  rad_ks.setConstant(1.0f);

  Cx4 out;
  std::unique_ptr<GridBase> gridder;
  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    gridder = make_grid_basis(kernel.get(), mapping, info.channels, basis, core.fast);
  } else {
    gridder = make_grid(kernel.get(), mapping, info.channels, core.fast);
  }
  auto const sdc = SDC::Choose(sdcOpts, traj, core.osamp.Get());
  out = gridder->Adj(sdc->Adj(rad_ks)).chip<0>(0);
  auto const fname = OutName(core.iname.Get(), core.oname.Get(), "traj", "h5");
  HD5::Writer writer(fname);
  writer.writeTensor(
    Cx5(out.reshape(Sz5{out.dimension(0), out.dimension(1), out.dimension(2), out.dimension(3), 1})), "traj-image");

  if (savePSF) {
    Log::Print(FMT_STRING("Calculating PSF"));
    auto const fft = FFT::Make<4, 3>(out.dimensions());
    fft->reverse(out);
    writer.writeTensor(
      Cx5(out.reshape(Sz5{out.dimension(0), out.dimension(1), out.dimension(2), out.dimension(3), 1})), "psf-image");
  }

  return EXIT_SUCCESS;
}
