#include "../../src/op/sense.h"
#include "../../src/tensorOps.h"
#include <catch2/catch.hpp>

TEST_CASE("ops-sense", "[ops]")
{
  long const channels = 2, mapSz = 4, gridSz = 6;

  // With credit to PyLops
  SECTION("Dot Test")
  {
    Cx3 x(mapSz, mapSz, mapSz), u(mapSz, mapSz, mapSz);
    Cx4 maps(channels, mapSz, mapSz, mapSz), y(channels, gridSz, gridSz, gridSz),
        v(channels, gridSz, gridSz, gridSz);

    v.setRandom();
    u.setRandom();
    // The maps need to be normalized for the Dot test
    maps.setRandom();
    Cx3 rss = ConjugateSum(maps, maps).sqrt();
    maps = maps / Tile(rss, channels);

    SenseOp sense(maps, y.dimensions());
    sense.A(u, y);
    sense.Adj(v, x);

    auto const yy = Dot(y, v);
    auto const xx = Dot(u, x);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}