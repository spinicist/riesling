#include "../../src/op/pad.hpp"
#include "../../src/tensorOps.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("ops-pad", "[ops]")
{
  Index const fullSz = 16;

  SECTION("Dot Test")
  {
    Index const cropSz = 7;
    Cx3 y(fullSz, fullSz, fullSz), yx(cropSz, cropSz, cropSz), x(cropSz, cropSz, cropSz),
      xy(fullSz, fullSz, fullSz);

    x.setRandom();
    y.setRandom();

    PadOp<3> crop(x.dimensions(), y.dimensions());
    crop.A(x, xy);
    crop.Adj(y, yx);

    // Don't forget conjugate on second dot argument
    auto const xx = Dot(x, yx);
    auto const yy = Dot(xy, y);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}