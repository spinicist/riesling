#include "../src/kernel.h"
#include <catch2/catch.hpp>

TEST_CASE("KB", "[KERNEL]")
{
  SECTION("3x3")
  {
    Kernel<3, 3> kb(2.f);
    auto const k = kb(Point3{0.f, 0.f, 0.f}, 1.f);
    fmt::print("k\n{}\n", k);
    CHECK(k(1, 1, 1) == 1.0f);
  }

  SECTION("5x5")
  {
    Kernel<5, 5> kb(2.f);
    auto const k = kb(Point3{0.f, 0.f, 0.f}, 1.f);
    fmt::print("k\n{}\n", k);
    CHECK(k(2, 2, 2) == 1.0f);
  }
}
