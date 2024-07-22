#include "types.hpp"

#include "basis/bernstein.hpp"
#include "io/writer.hpp"
#include "log.hpp"
#include "inputs.hpp"

void main_bernstein(args::Subparser &parser)
{
  args::Positional<std::string> oname(parser, "OUTPUT", "Name for the basis file");

  args::ValueFlag<Index> traces(parser, "T", "Number of traces (1)", {"traces", 't'}, 1);
  args::ValueFlag<Index> order(parser, "N", "Polynomial order", {"order", 'N'}, 4);
  ParseCommand(parser, oname);

  auto b = rl::BernsteinPolynomial(order.Get(), traces.Get());

  rl::HD5::Writer writer(oname.Get());
  writer.writeTensor(rl::HD5::Keys::Basis, b.B.dimensions(), b.B.data(), rl::HD5::Dims::Basis);
}
