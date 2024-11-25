#include "rl/op/sense.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("SENSE", "[SENSE]")
{
  Index const channels = 2, mapSz = 4;
  // With credit to PyLops
  SECTION("Dot Test")
  {
    Cx4 x(1, mapSz, mapSz, mapSz), u(1, mapSz, mapSz, mapSz);
    Cx5 maps(1, channels, mapSz, mapSz, mapSz);
    Cx5 y(1, channels, mapSz, mapSz, mapSz), v(1, channels, mapSz, mapSz, mapSz);

    v.setRandom();
    u.setRandom();
    // The maps need to be normalized for the Dot test
    maps.setRandom();
    maps =
      maps / DimDot<1>(maps, maps).sqrt().reshape(Sz5{1, 1, mapSz, mapSz, mapSz}).broadcast(Sz5{1, channels, 1, 1, 1});

    TOps::SENSE sense(maps, false, 1);
    y = sense.forward(u);
    x = sense.adjoint(v);

    auto const yy = Dot(y, v);
    auto const xx = Dot(u, x);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}
