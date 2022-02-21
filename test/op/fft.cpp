#include "../../src/op/fft.hpp"
#include "../../src/tensorOps.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("ops-fft")
{
  Index const sz = 16;
  Cx5 ws(sz, sz, sz, sz, sz);
  auto const dims = ws.dimensions();
  FFTOp<5> fft(ws);

  SECTION("FFT-Dot")
  {
    Cx5 y(dims), yx(dims), x(dims), xy(dims);

    x.setRandom();
    y.setRandom();

    xy = fft.A(x);
    yx = fft.Adj(y);

    // Don't forget conjugate on second dot argument
    auto const xx = Dot(x, yx);
    auto const yy = Dot(xy, y);
    CHECK(std::abs((yy - xx) / (yy + xx + 1.e-15f)) == Approx(0).margin(1.e-6));
  }
}