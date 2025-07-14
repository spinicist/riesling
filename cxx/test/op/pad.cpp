#include "rl/op/pad.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("pad", "[op]")
{
  Index const fullSz = 6;

  SECTION("Dot Test")
  {
    Index const cropSz = 3;
    Cx3 y(fullSz, fullSz, fullSz), yx(cropSz, cropSz, cropSz), x(cropSz, cropSz, cropSz), xy(fullSz, fullSz, fullSz);

    x.setRandom();
    y.setRandom();

    TOps::Pad<3> pad(x.dimensions(), y.dimensions());
    xy = pad.forward(x);
    yx = pad.adjoint(y);

    INFO("xy\n" << xy);

    CHECK(std::abs(xy(0, 0, 0)) == 0.f);
    CHECK(std::abs(xy(fullSz - 1, fullSz - 1, fullSz - 1)) == 0.f);

    Index lC = (fullSz / 2) - (cropSz / 2);
    Index rC = (fullSz / 2) + (cropSz / 2);
    CHECK(xy(lC, lC, lC) == x(0, 0, 0));
    CHECK(xy(rC, rC, rC) == x(cropSz - 1, cropSz - 1, cropSz - 1));

    // Don't forget conjugate on second dot argument
    auto const xx = Dot<false>(x, yx);
    auto const yy = Dot<false>(xy, y);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}