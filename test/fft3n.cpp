#include "../src/fft_many.h"
#include "../src/log.h"
#include "../src/tensorOps.h"
#include <catch2/catch.hpp>
#include <fmt/format.h>

TEST_CASE("FFT-Many", "[FFT::Many]")
{
  Log log;
  FFT::Start(log);

  auto sx = GENERATE(3, 5, 7, 16);
  auto sy = sx;
  SECTION("FFT::Many<4>")
  {
    auto sz = GENERATE(1, 3, 7, 8, 16);
    long const N = sx * sy * sz;
    long const nc = 32;
    Cx4 data(nc, sx, sy, sz);
    Cx4 ref(nc, sx, sy, sz);
    FFT::Many<4> fft(data, log);

    ref.setConstant(1.f);
    data.setZero();
    data.chip(sz / 2, 3).chip(sy / 2, 2).chip(sx / 2, 1).setConstant(sqrt(N));
    fft.forward();
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-6f * N * nc));
    fft.reverse();
    ref.setZero();
    ref.chip(sz / 2, 3).chip(sy / 2, 2).chip(sx / 2, 1).setConstant(sqrt(N));
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-6f * N * nc));
  }

  FFT::End(log);
}
