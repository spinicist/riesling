#include "types.hpp"

#include "basis/basis.hpp"
#include "inputs.hpp"
#include "io/hd5.hpp"
#include "log.hpp"
#include "tensors.hpp"

using namespace rl;

void main_basis_outer(args::Subparser &parser)
{
  args::Positional<std::string> iname1(parser, "F", "First file");
  args::Positional<std::string> iname2(parser, "F", "Second file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Output file");

  ParseCommand(parser);
  if (!iname1) { throw args::Error("Input file 1 not specified"); }
  if (!iname2) { throw args::Error("Input file 2 not specified"); }
  if (!oname) { throw args::Error("Output file not specified"); }

  auto b1 = LoadBasis(iname1.Get());
  auto b2 = LoadBasis(iname2.Get());

  if (b1->nTrace() != 1) { Log::Fail("Expected 1st basis to have 1 trace"); }
  if (b2->nSample() != 1) { Log::Fail("Expected 2nd basis to have 1 sample"); }

  auto const n1 = b1->nB();
  auto const n2 = b2->nB();
  auto const nS = b1->nSample();
  auto const nT = b2->nTrace();
  Cx3        nb(n1 * n2, nS, nT);

  for (Index i2 = 0; i2 < n2; i2++) {
    for (Index i1 = 0; i1 < n1; i1++) {
      for (Index is = 0; is < nS; is++) {
        for (Index it = 0; it < nT; it++) {
          nb(i2 * n1 + i1, is, it) = b1->B(i1, is, 0) * b2->B(i2, 0, it);
        }
      }
    }
  }

  Basis const b3(nb);
  b3.write(oname.Get());
}
