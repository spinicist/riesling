#include "rl/fft.hpp"
#include "rl/log/log.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("FFT1", "[FFT]")
{
  auto N = GENERATE(6, 8, 10, 12);
  Cx1  data(N);
  Cx1  ref(data.dimensions());
  SECTION("Frequency Impulse")
  {
    data.setConstant(1.f);
    ref.setZero();
    ref(N / 2) = sqrt(N); // Parseval's theorem
    INFO("Ref " << ref << "\nData " << data);
    FFT::Forward(data);
    INFO("Ref " << ref << "\nData " << data);
    CHECK(Norm<false>(data - ref) == Approx(0.f).margin(1.e-3f));
    FFT::Adjoint(data);
    INFO("Ref " << ref << "\nData " << data);
    ref.setConstant(1.f);
    CHECK(Norm<false>(data - ref) == Approx(0.f).margin(1.e-3f));
  }

  SECTION("Spatial Impulse Left")
  {
    ref.setConstant(1.f);
    for (Index ii = 0; ii < N; ii++) {
      if (ii % 2 != (N / 2) % 2) { ref(ii) = -1.f; }
    }
    data.setZero();
    data(0) = sqrt(N); // Parseval's theorem
    INFO("Before " << data);
    FFT::Forward(data);
    INFO("After  " << data << "\nRef    " << ref << "\n");
    CHECK(Norm<false>(data - ref) == Approx(0.f).margin(1.e-3f));

    INFO("Before " << data);
    FFT::Adjoint(data);
    INFO("After  " << data << "\nRef    " << ref << "\n");
    ref.setZero();
    ref(0) = sqrt(N); // Parseval's theorem
    CHECK(Norm<false>(data - ref) == Approx(0.f).margin(1.e-3f));
  }

  SECTION("Spatial Impulse Mid")
  {
    ref.setConstant(1.f);
    data.setZero();
    data(N / 2) = sqrt(N); // Parseval's theorem
    INFO("Before " << data);
    FFT::Forward(data);
    INFO("After  " << data << "\nRef    " << ref << "\n");
    CHECK(Norm<false>(data - ref) == Approx(0.f).margin(1.e-3f));

    INFO("Before " << data);
    FFT::Adjoint(data);
    INFO("After  " << data << "\nRef    " << ref << "\n");
    ref.setZero();
    ref(N / 2) = sqrt(N); // Parseval's theorem
    CHECK(Norm<false>(data - ref) == Approx(0.f).margin(1.e-3f));
  }
}
