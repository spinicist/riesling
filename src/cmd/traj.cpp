#include "types.hpp"

#include "fft/fft.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "op/make_grid.hpp"
#include "parse_args.hpp"
#include "sdc.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"
#include <complex>

using namespace rl;

int main_traj(args::Subparser &parser)
{
  CoreOpts coreOpts(parser);
  SDC::Opts sdcOpts(parser);

  args::Flag savePSF(parser, "PSF", "Write out Point-Spread-Function", {"psf", 'p'});

  ParseCommand(parser, coreOpts.iname);

  HD5::Reader reader(coreOpts.iname.Get());
  Trajectory traj(reader, coreOpts.frames.Get());
  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto gridder = make_grid<Cx, 3>(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), 1, basis);
  auto const sdc = SDC::Choose(sdcOpts, traj, 1, coreOpts.ktype.Get(), coreOpts.osamp.Get());
  Cx3 rad_ks(1, traj.nSamples(), traj.nTraces());
  rad_ks.setConstant(1.0f);
  (*sdc)(rad_ks, rad_ks);
  Cx5 out = gridder->adjoint(rad_ks);
  auto const fname = OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "traj", "h5");
  HD5::Writer writer(fname);
  writer.writeTensor(out, "traj-image");

  if (savePSF) {
    Log::Print(FMT_STRING("Calculating PSF"));
    auto const fft = FFT::Make<5, 3>(out.dimensions());
    fft->reverse(out);
    writer.writeTensor(out, "psf-image");
  }

  return EXIT_SUCCESS;
}
