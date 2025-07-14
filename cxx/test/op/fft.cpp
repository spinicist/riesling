#include "rl/op/fft.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("fft", "[op]")
{
  Index const      sz = 16;
  Sz5 const        dims{sz, sz, sz, sz, sz};
  TOps::FFT<5, 3> fft(dims);

  SECTION("FFT-Dot")
  {
    Cx5 y(dims), yx(dims), x(dims), xy(dims);

    x.setRandom();
    y.setRandom();

    xy = fft.forward(x);
    yx = fft.adjoint(y);

    // Don't forget conjugate on second dot argument
    auto const xx = Dot<false>(x, yx);
    auto const yy = Dot<false>(xy, y);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}