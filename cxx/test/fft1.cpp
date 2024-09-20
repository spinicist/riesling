#include "fft.hpp"
#include "log.hpp"
#include "tensors.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_approx.hpp>
#include <fmt/format.h>

using namespace rl;
using namespace Catch;

TEST_CASE("FFT1","[FFT]")
{
  auto N = GENERATE(8);
  SECTION("<1>")
  {
    Cx1 data(N);
    Cx1 ref(data.dimensions());

    ref.setConstant(1.f);
    data.setZero();
    data(N / 2) = sqrt(N); // Parseval's theorem
    INFO("Ref " << ref << "\nData " << data);
    FFT::Forward(data);
    INFO("Ref " << ref << "\nData " << data);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-3f));
    FFT::Adjoint(data);
    INFO("Ref " << ref << "\nData " << data);
    ref.setZero();
    ref(N / 2) = sqrt(N);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-3f));
  }
}
