#include "algo/decomp.h"
#include "tensorOps.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

using namespace rl;

TEST_CASE("decomp")
{
  Index const nvar = 64;
  Index const nsamp = 256;

  SECTION("PCA")
  {
    // Due to how channels are stored, we put each sample in a column instead of a row
    Cx2 data(nvar, nsamp);
    data.setRandom();
    PCA(CollapseToConstMatrix(data), 1);
  }
}