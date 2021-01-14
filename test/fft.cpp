#include "../src/fft3.h"
#include "../src/log.h"
#include <catch2/catch.hpp>
#include <fmt/format.h>

TEST_CASE("FFT Basic Sanity", "[FFT]")
{
  Log log(false);

  SECTION("3D-Odd")
  {
    long const sx = 3;
    long const sy = 3;
    long const sz = 3;
    long const N = sx * sy * sz;
    Cx3 data(Sz3{sx, sy, sz});
    Cx3 ref(sx, sy, sz);
    FFT3 fft(data, log);

    data.setConstant(1.f);
    ref.setZero();
    ref(sx / 2, sy / 2, sz / 2) = sqrt(N);
    fft.forward();
    CHECK(norm(data - ref) == Approx(0.f).margin(1.e-6f)); // Parseval's theorem
    fft.reverse();
    ref.setConstant(1.f);
    CHECK(norm(data - ref) == Approx(0.f).margin(1.e-6f));
  }

  SECTION("3D-Even")
  {
    long const sx = 16;
    long const sy = 16;
    long const sz = 16;
    long const N = sx * sy * sz;
    Cx3 data(sx, sy, sz);
    Cx3 ref(sx, sy, sz);
    FFT3 fft(data, log);

    data.setConstant(1.f);
    ref.setZero();
    ref(sx / 2, sy / 2, sz / 2) = sqrt(N);
    fft.forward();
    CHECK(norm(data - ref) == Approx(0.f).margin(1.e-6f)); // Parseval's theorem
    fft.reverse();
    ref.setConstant(1.f);
    CHECK(norm(data - ref) == Approx(0.f).margin(1.e-6f));
  }
}
