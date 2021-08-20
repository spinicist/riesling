#include "../../src/op/crop.h"
#include "../../src/tensorOps.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("ops-crop", "[ops]")
{
  long const fullSz = 16;

  SECTION("Dot Test")
  {
    long const cropSz = 8;
    Cx3 x(fullSz, fullSz, fullSz), xy(cropSz, cropSz, cropSz), y(cropSz, cropSz, cropSz),
        yx(fullSz, fullSz, fullSz);

    x.setRandom();
    y.setRandom();

    Crop3 crop(x.dimensions(), y.dimensions());
    crop.A(x, xy);
    crop.Adj(y, yx);

    CHECK(Dot(x, yx) == Dot(xy, y));
  }
}