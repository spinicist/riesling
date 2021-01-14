#include "../src/fft3.h"
#include "../src/kernel_kb.h"
#include "../src/tensorOps.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("3x3", "[KB3]")
{
  long const sz = 3;
  KaiserBessel kernel(sz, 1);
  Log nullLog(false);
  Cx3 temp(sz, sz, sz);
  FFT3 fft(temp, nullLog);
  temp.setZero();
  SECTION("Start/Size")
  {
    CHECK(kernel.start()[0] == -sz / 2);
    CHECK(kernel.start()[1] == -sz / 2);
    CHECK(kernel.start()[2] == -sz / 2);
    CHECK(kernel.size()[0] == sz);
    CHECK(kernel.size()[1] == sz);
    CHECK(kernel.size()[2] == sz);
  }

  SECTION("Centered")
  {
    // This test is incomplete
    Cx3 const k = kernel.kspace(Point3::Zero());
    CHECK(Sum(k.real()) == Approx(1.f));
    Cx3 img = kernel.image(Point3::Zero(), temp.dimensions());
    // CHECK(Norm(img) == Approx(1.f));
    temp = k.cast<std::complex<float>>();
    fft.reverse();
    // CHECK(Norm(k - img.abs()) == Approx(0.f));
  }
}
