#include "algo/decomp.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("decomp")
{
  Index const nvar = 64;
  Index const nsamp = 256;

  SECTION("PCA")
  {
    // Due to how channels are stored, we put each sample in a column instead of a row
    Cx2 data(nvar, nsamp);
    data.setRandom();
    Cx2 vecs(nvar, nvar);
    R1 vals(nvar);
    PCA(data, vecs, vals);
  }
}