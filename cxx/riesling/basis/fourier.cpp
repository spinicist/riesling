#include "rl/basis/fourier.hpp"
#include "rl/algo/gs.hpp"
#include "rl/io/writer.hpp"
#include "rl/log/log.hpp"
#include "rl/types.hpp"

#include "inputs.hpp"

#include <complex>
#include <numbers>

using namespace std::literals::complex_literals;

void main_basis_fourier(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> N(parser, "N", "Number of Fourier harmonics (4)", {"N", 'N'}, 4);
  args::ValueFlag<Index> samples(parser, "S", "Number of samples (1)", {"samples", 's'}, 1);
  args::ValueFlag<Index> traces(parser, "T", "Number of traces (1)", {"traces", 't'}, 1);
  args::ValueFlag<Index> osamp(parser, "O", "Oversampling (1)", {"osamp", 'o'}, 1.f);
  ParseCommand(parser, oname);

  rl::FourierBasis fb(N.Get(), samples.Get(), traces.Get(), osamp.Get());
  rl::HD5::Writer  writer(oname.Get());
  writer.writeTensor(rl::HD5::Keys::Basis, fb.basis.dimensions(), fb.basis.data(), rl::HD5::Dims::Basis);
}
