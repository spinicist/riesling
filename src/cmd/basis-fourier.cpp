#include "types.hpp"

#include "basis/fourier.hpp"
#include "log.hpp"
#include "parse_args.hpp"

#include <unsupported/Eigen/Splines>

int main_basis_fourier(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index>  N(parser, "N", "Number of Fourier harmonics (4)", {"n", 'n'}, 4);
  args::ValueFlag<Index> samples(parser, "S", "Number of samples (1)", {"samples", 's'}, 1);
  args::ValueFlag<Index> traces(parser, "T", "Number of traces (1)", {"traces"}, 1);

  ParseCommand(parser, oname);

  rl::FourierBasis fb(N.Get(), samples.Get(), traces.Get());
  fb.writeTo(oname.Get());

  return EXIT_SUCCESS;
}
