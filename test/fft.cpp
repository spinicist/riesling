#include "../src/fft.h"
#include "../src/log.h"
#include <catch2/catch.hpp>
#include <fmt/format.h>

TEST_CASE("FFT Basic Sanity", "[FFT]")
{
  Log log(false);

  SECTION("3D Single")
  {
    long const sz = 32;
    long const N = sz * sz * sz;
    Cx3 cube(Sz3{sz, sz, sz});
    FFT3 fft(cube, log);

    cube.setConstant(1.f);
    fft.forward();
    CHECK(norm(cube) == Approx(sqrt(N))); // Parseval's theorem
    CHECK(std::abs(cube(0, 0, 0)) == Approx(sqrt(N)));
    CHECK(std::abs(cube(0, 0, 1)) == Approx(0.f));
    CHECK(std::abs(cube(0, 1, 0)) == Approx(0.f));
  }
}
