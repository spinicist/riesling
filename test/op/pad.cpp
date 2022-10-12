#include "../../src/op/pad.hpp"
#include "../../src/tensorOps.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/ostream.h>

using namespace rl;
using namespace Catch;

TEST_CASE("ops-pad")
{
  Index const fullSz = 16;

  SECTION("Dot Test")
  {
    Index const cropSz = 7;
    Cx3 y(fullSz, fullSz, fullSz), yx(cropSz, cropSz, cropSz), x(cropSz, cropSz, cropSz), xy(fullSz, fullSz, fullSz);

    x.setRandom();
    y.setRandom();

    PadOp<3> crop(x.dimensions(), y.dimensions());
    xy = crop.forward(x);
    yx = crop.adjoint(y);

    // Don't forget conjugate on second dot argument
    auto const xx = Dot(x, yx);
    auto const yy = Dot(xy, y);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}