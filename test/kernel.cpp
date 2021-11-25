#include "../src/kernel.h"
#include "../src/tensorOps.h"
#include <catch2/catch.hpp>

TEST_CASE("Kernels", "[KERNELS]")
{
  SECTION("KB3")
  {
    KaiserBessel<3, 3> kb(2.f);
    auto const k = kb(Point3{0.f, 0.f, 0.f});
    CHECK(k(1, 1, 1) == Approx(0.37933f).margin(1.e-5));
    CHECK(Sum(k) == Approx(1.f).margin(1.e-9));
  }

  SECTION("KB5")
  {
    KaiserBessel<5, 5> kb(2.f);
    auto const k = kb(Point3{0.f, 0.f, 0.f});
    CHECK(k(2, 2, 2) == Approx(0.17432f).margin(1.e-5));
    CHECK(Sum(k) == Approx(1.f).margin(1.e-9));
  }

  SECTION("Pipe")
  {
    PipeSDC<5, 5> pipe(2.1f);
    auto const k = pipe(Point3(0.f, 0.f, 0.f));
    CHECK(Sum(k) == Approx(1.f).margin(1.e-3));
  }
}
