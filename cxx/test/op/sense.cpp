#include "rl/op/sense.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("SENSE", "[op]")
{
  Index const channels = 2, mapSz = 4;
  // With credit to PyLops
  SECTION("Dot Test")
  {
    Cx4 x(mapSz, mapSz, mapSz, 1), u(mapSz, mapSz, mapSz, 1);
    Cx5 maps(mapSz, mapSz, mapSz, 1, channels);
    Cx5 y(mapSz, mapSz, mapSz, 1, channels), v(mapSz, mapSz, mapSz, 1, channels);

    v.setRandom();
    u.setRandom();
    // The maps need to be normalized for the Dot test
    maps.setRandom();
    maps = maps / DimDot<4>(maps, maps).sqrt().reshape(Sz5{mapSz, mapSz, mapSz, 1, 1}).broadcast(Sz5{1, 1, 1, 1, channels});

    TOps::SENSEOp sense(maps, 1);
    y = sense.forward(u);
    x = sense.adjoint(v);

    auto const yy = Dot<false>(y, v);
    auto const xx = Dot<false>(u, x);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}
