#include "../src/fft1.h"
#include "../src/log.h"
#include "../src/tensorOps.h"
#include <catch2/catch.hpp>
#include <fmt/format.h>

TEST_CASE("1D-FFT", "[FFT1]")
{
  Log log;
  FFT::Start(log);

  auto N = GENERATE(4, 8, 32);
  SECTION("FFT1")
  {
    FFT1DReal2Complex fft(N, log);

    R1 realRef(N);
    realRef.setZero();
    realRef(N / 2) = sqrt(N); // Parseval's theorem
    Cx1 complexRef(N / 2 + 1);
    complexRef.setConstant(1.f);

    Cx1 complex = fft.forward(realRef);
    CHECK(Norm(complexRef - complex) == Approx(0.f).margin(1.e-3f));
    R1 real = fft.reverse(complex);
    CHECK(Norm(real - realRef) == Approx(0.f).margin(1.e-3f));
  }
  FFT::End(log);
}
