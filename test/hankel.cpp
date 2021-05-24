#include "../src/hankel.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("Hankel", "[Hankel]")
{
  Log log;

  long const nchan = 2;
  long const gridSz = 10;
  Cx4 grid(nchan, gridSz, gridSz, gridSz);
  grid.chip(0, 0).setConstant(1.f);
  grid.chip(1, 0).setConstant(2.f);
  long const kRad = 2;
  long const calRad = 4;

  SECTION("No Gap")
  {
    Cx5 k = ToKernels(grid, kRad, calRad, 0, log);

    CHECK(k.dimension(0) == nchan);
    long const kW = 2 * kRad - 1;
    CHECK(k.dimension(1) == kW);
    CHECK(k.dimension(2) == kW);
    CHECK(k.dimension(3) == kW);
    long const calW = 2 * calRad - 1;
    CHECK(k.dimension(4) == (calW * calW * calW));
    CHECK(k(0, 0, 0, 0, 0) == 1.f);
    CHECK(k(1, 0, 0, 0, 0) == 2.f);
  }

  SECTION("Gap")
  {
    long const gap = 1;
    grid.chip(gridSz / 2, 3).chip(gridSz / 2, 2).chip(gridSz / 2, 1).setZero();
    Cx5 k = ToKernels(grid, kRad, calRad, gap, log);
    CHECK(k.dimension(0) == nchan);
    long const kW = 2 * kRad - 1;
    CHECK(k.dimension(1) == kW);
    CHECK(k.dimension(2) == kW);
    CHECK(k.dimension(3) == kW);
    long const calW = 2 * calRad - 1;
    long const gapW = ((gap + kRad) * 2) - 1;
    CHECK(k.dimension(4) == (calW * calW * calW) - (gapW * gapW * gapW));
    CHECK(k(0, 0, 0, 0, 0) == 1.f);
    CHECK(k(1, 0, 0, 0, 0) == 2.f);
    CHECK(B0((k.real() > k.real().constant(0.f)).all())());
  }
}