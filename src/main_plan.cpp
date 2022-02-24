#include "types.h"

#include "fft/fft.hpp"
#include "io.h"
#include "log.h"
#include "op/grid.h"
#include "parse_args.h"

int main_plan(args::Subparser &parser)
{
  COMMON_RECON_ARGS;
  args::ValueFlag<double> timelimit(
    parser, "LIMIT", "Time limit for FFT planning (default 60 s)", {"time", 't'}, 60.0);
  args::ValueFlag<std::string> basisFile(
    parser, "BASIS", "Read subspace basis from .h5 file", {"basis", 'b'});

  ParseCommand(parser, iname);
  FFT::Start();
  FFT::SetTimelimit(timelimit.Get());
  HD5::RieslingReader reader(iname.Get());
  auto const traj = reader.trajectory();
  auto const info = traj.info();
  auto const kernel = make_kernel(ktype.Get(), info.type, osamp.Get());
  auto gridder = make_grid(kernel.get(), traj.mapping(kernel->inPlane(), osamp.Get()), fastgrid);
  auto const fftN = FFT::Make<5, 3>(gridder->inputDimensions());
  auto grid1 = make_grid(kernel.get(), traj.mapping(kernel->inPlane(), osamp.Get(), 1), fastgrid);
  auto const fft1 = FFT::Make<5, 3>(grid1->inputDimensions());

  if (basisFile) {
    HD5::Reader basisReader(basisFile.Get());
    R2 basis = basisReader.readTensor<R2>(HD5::Keys::Basis);
    auto gb = make_grid_basis(kernel.get(), gridder->mapping(), basis, fastgrid);
    auto const fftB = FFT::Make<5, 3>(gb->inputDimensions());
  }

  FFT::End();
  return EXIT_SUCCESS;
}
