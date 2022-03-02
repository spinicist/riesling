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
  SECTION("KB31")
  {
    auto const kb = KaiserBessel<3, 1>(2.f);
    auto const k = kb.k(Point3{0.f, 0.f, 0.f});
    CHECK(k(1, 1, 0) == Approx(0.96328f).margin(1.e-5));
    CHECK(Sum(k) == Approx(1.f).margin(1.e-9));
  }
  SECTION("KB51")
  {
    auto const kb = KaiserBessel<5, 1>(2.f);
    auto const k = kb.k(Point3{0.f, 0.f, 0.f});
    CHECK(k(1, 1, 0) == Approx(0.01997f).margin(1.e-5));
    CHECK(Sum(k) == Approx(1.f).margin(1.e-9));
  }
  SECTION("FI31")
  {
    auto const fi = FlatIron<3, 1>(2.f);
    auto const k = fi.k(Point3{0.f, 0.f, 0.f});
    CHECK(k(1, 1, 0) == Approx(0.99609f).margin(1.e-5));
    CHECK(Sum(k) == Approx(1.f).margin(1.e-9));
  }
  SECTION("KB3")
  {
    auto const kb = KaiserBessel<3, 3>(2.f);
    auto const k = kb.k(Point3{0.f, 0.f, 0.f});
    CHECK(k(1, 1, 1) == Approx(0.94592f).margin(1.e-5));
    CHECK(Sum(k) == Approx(1.f).margin(1.e-9));
  }

  SECTION("KB5")
  {
    auto const kb = KaiserBessel<5, 5>(2.f);
    auto const k = kb.k(Point3{0.f, 0.f, 0.f});
    CHECK(k(2, 2, 2) == Approx(0.34064f).margin(1.e-5));
    CHECK(Sum(k) == Approx(1.f).margin(1.e-9));
  }

  SECTION("Pipe51")
  {
    auto const pipe = PipeSDC<5, 1>(2.f);
    auto const k = pipe.k(Point3{0.f, 0.f, 0.f});
    CHECK(Sum(k) == Approx(1.f).margin(1.e-3));
    fmt::print("{}\n", k);
  }
  SECTION("Pipe55")
  {
    auto const pipe = PipeSDC<5, 5>(2.f);
    auto const k = pipe.k(Point3{0.f, 0.f, 0.f});
    CHECK(Sum(k) == Approx(1.f).margin(1.e-3));
    fmt::print("{}\n", k);
  }
}
