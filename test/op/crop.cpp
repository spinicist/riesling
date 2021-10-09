#include "../../src/op/crop.h"
#include "../../src/tensorOps.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("ops-crop", "[ops]")
{
  long const fullSz = 16;

  SECTION("Dot Test")
  {
    long const cropSz = 7;
    Cx3 x(fullSz, fullSz, fullSz), xy(cropSz, cropSz, cropSz), y(cropSz, cropSz, cropSz),
        yx(fullSz, fullSz, fullSz);

    x.setRandom();
    y.setRandom();

    CropOp3 crop(x.dimensions(), y.dimensions());
    crop.A(x, xy);
    crop.Adj(y, yx);

    // Don't forget conjugate on second dot argument
    auto const xx = Dot(x, yx);
    auto const yy = Dot(xy, y);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}