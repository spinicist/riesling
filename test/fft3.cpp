#include "../src/fft3.h"
#include "../src/log.h"
#include "../src/tensorOps.h"
#include <catch2/catch.hpp>
#include <fmt/format.h>

TEST_CASE("3D-FFT", "[FFT3]")
{
  Log log;
  FFT::Start(log);

  auto sx = GENERATE(3, 5, 7, 32);
  auto sy = sx;
  SECTION("FFT3")
  {
    auto sz = GENERATE(1, 2, 3, 16, 32);
    long const N = sx * sy * sz;
    Cx3 data(Sz3{sx, sy, sz});
    Cx3 ref(sx, sy, sz);
    FFT3 fft(data, log);

    ref.setConstant(1.f);
    data.setZero();
    data(sx / 2, sy / 2, sz / 2) = sqrt(N); // Parseval's theorem
    fft.forward(data);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-3f));
    fft.reverse(data);
    ref.setZero();
    ref(sx / 2, sy / 2, sz / 2) = sqrt(N);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-3f));
  }
  FFT::End(log);
}
