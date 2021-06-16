#include "../src/cropper.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("Cropper", "[Crop]")
{
  Log log;
  long const fullSz = 16;
  R3 grid(fullSz, fullSz, fullSz);
  grid.setZero();
  grid(fullSz / 2, fullSz / 2, fullSz / 2) = 1.f;

  SECTION("Even")
  {
    long const small = 8;
    Cropper crop(Sz3{fullSz, fullSz, fullSz}, Sz3{small, small, small}, log);
    R3 const cropSmall = crop.crop3(grid);

    CHECK(cropSmall.dimension(0) == small);
    CHECK(cropSmall.dimension(1) == small);
    CHECK(cropSmall.dimension(2) == small);

    CHECK(cropSmall(small / 2, small / 2, small / 2) == 1.f);
    CHECK(cropSmall(small / 2 - 1, small / 2 - 1, small / 2 - 1) == 0.f);
  }

  SECTION("Odd")
  {
    long const small = 7;
    Cropper crop(Sz3{fullSz, fullSz, fullSz}, Sz3{small, small, small}, log);
    R3 const cropSmall = crop.crop3(grid);

    CHECK(cropSmall.dimension(0) == small);
    CHECK(cropSmall.dimension(1) == small);
    CHECK(cropSmall.dimension(2) == small);

    CHECK(cropSmall(small / 2, small / 2, small / 2) == 1.f);
    CHECK(cropSmall(small / 2 - 1, small / 2 - 1, small / 2 - 1) == 0.f);
  }
}