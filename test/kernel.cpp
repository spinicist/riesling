#include "../src/kernel.hpp"
#include "../src/tensorOps.h"
#include <catch2/catch.hpp>

TEST_CASE("kernels")
{
  SECTION("NN")
  {
    auto const nn = NearestNeighbour();
    auto const k = nn.k(Point3{0.f, 0.f, 0.f});
    CHECK(k(0, 0, 0) == Approx(1.f).margin(1.e-5));
    CHECK(Sum(k) == Approx(1.f).margin(1.e-9));
  }
  SECTION("KB3")
  {
    auto const kb = KaiserBessel<3, 3>(2.f);
    auto const k = kb.k(Point3{0.f, 0.f, 0.f});
    CHECK(k(1, 1, 1) == Approx(0.37933f).margin(1.e-5));
    CHECK(Sum(k) == Approx(1.f).margin(1.e-9));
  }

  SECTION("KB5")
  {
    auto const kb = KaiserBessel<5, 5>(2.f);
    auto const k = kb.k(Point3{0.f, 0.f, 0.f});
    CHECK(k(2, 2, 2) == Approx(0.17432f).margin(1.e-5));
    CHECK(Sum(k) == Approx(1.f).margin(1.e-9));
  }

  SECTION("Pipe")
  {
    auto const pipe = PipeSDC<5, 5>(2.f);
    auto const k = pipe.k(Point3{0.f, 0.f, 0.f});
    CHECK(Sum(k) == Approx(1.f).margin(1.e-3));
  }
}
