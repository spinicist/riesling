#include "../../src/op/sense.hpp"
#include "../../src/tensorOps.h"
#include <catch2/catch.hpp>

TEST_CASE("ops-sense", "[ops]")
{
  Index const channels = 2, mapSz = 4, gridSz = 6;
  Log log;
  // With credit to PyLops
  SECTION("Dot Test")
  {
    Cx4 x(1, mapSz, mapSz, mapSz), u(1, mapSz, mapSz, mapSz);
    Cx4 maps(channels, mapSz, mapSz, mapSz);
    Cx5 y(channels, 1, gridSz, gridSz, gridSz), v(channels, 1, gridSz, gridSz, gridSz);

    v.setRandom();
    u.setRandom();
    // The maps need to be normalized for the Dot test
    maps.setRandom();
    Cx3 rss = ConjugateSum(maps, maps).sqrt();
    maps = maps / Tile(rss, channels);

    SenseOp sense(maps, y.dimensions(), log);
    sense.A(u, y);
    sense.Adj(v, x);

    auto const yy = Dot(y, v);
    auto const xx = Dot(u, x);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}