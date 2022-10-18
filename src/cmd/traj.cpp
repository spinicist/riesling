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
  Trajectory traj(reader);
  auto const basis = ReadBasis(coreOpts.basisFile.Get());
  auto gridder = make_grid<Cx, 3>(traj, coreOpts.ktype.Get(), coreOpts.osamp.Get(), 1, basis);
  auto const sdc = SDC::Choose(sdcOpts, traj, 1, coreOpts.ktype.Get(), coreOpts.osamp.Get());
  Cx3 rad_ks(1, traj.nSamples(), traj.nTraces());
  rad_ks.setConstant(1.0f);
  Cx4 out = gridder->adjoint((*sdc)(rad_ks)).chip<0>(0);
  auto const fname = OutName(coreOpts.iname.Get(), coreOpts.oname.Get(), "traj", "h5");
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
