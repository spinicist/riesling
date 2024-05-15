#include "op/fft.hpp"
#include "log.hpp"
#include "tensors.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/ostream.h>

using namespace rl;
using namespace Catch;

TEST_CASE("ops-fft")
{
  Index const      sz = 16;
  Sz5 const        dims{sz, sz, sz, sz, sz};
  Ops::FFTOp<5, 3> fft(dims);

  SECTION("FFT-Dot")
  {
    Cx5 y(dims), yx(dims), x(dims), xy(dims);

    x.setRandom();
    y.setRandom();

    xy = fft.forward(x);
    yx = fft.adjoint(y);

    // Don't forget conjugate on second dot argument
    auto const xx = Dot(x, yx);
    auto const yy = Dot(xy, y);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}