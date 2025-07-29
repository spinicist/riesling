#include "rl/algo/gs.hpp"
#include "rl/fft.hpp"
#include "rl/io/writer.hpp"
#include "rl/log/log.hpp"
#include "rl/op/pad.hpp"
#include "rl/types.hpp"

#include "inputs.hpp"

#include <complex>
#include <numbers>

using namespace std::literals::complex_literals;
using namespace rl;

void main_basis_fourier(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> N(parser, "N", "Number of Fourier harmonics (4)", {"N", 'N'}, 4);
  args::ValueFlag<Index> P(parser, "P", "Number of points (128)", {"P", 'P'}, 128);
  args::Flag             traces(parser, "T", "Basis is along traces dimension", {"traces", 't'});
  args::ValueFlag<Index> osamp(parser, "O", "Oversampling (1)", {"osamp", 'o'}, 1.f);
  ParseCommand(parser, oname);

  Cx2 eye(N.Get(), N.Get());
  eye.setZero();
  float const energy = 1.f;
  for (Index ii = 0; ii < N.Get(); ii++) {
    eye(ii, ii) = std::polar<float>(energy, (ii - N.Get()/2) * M_PI);
  }
  Cx2 padded = TOps::Pad<2>(eye.dimensions(), Sz2{N.Get(), (Index)(osamp.Get() * P.Get())}).forward(eye);
  FFT::Adjoint(padded, Sz1{1});
  Cx2 basis = TOps::Pad<2>(Sz2{N.Get(), P.Get()}, padded.dimensions()).adjoint(padded);
  rl::HD5::Writer writer(oname.Get());
  Sz3 const       shape{N.Get(), traces ? 1 : P.Get(), traces ? P.Get() : 1};
  writer.writeTensor(rl::HD5::Keys::Basis, shape, basis.data(), rl::HD5::Dims::Basis);
}
