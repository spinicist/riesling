#include "../src/fft/fft.hpp"
#include "../src/log.hpp"
#include "../src/tensorOps.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_approx.hpp>
#include <fmt/format.h>

using namespace rl;
using namespace Catch;

TEST_CASE("FFT1","[FFT]")
{
  auto N = GENERATE(32);
  SECTION("<1>")
  {
    Cx1 data(N);
    Cx1 ref(data.dimensions());
    auto const fft = FFT::Make<1, 1>(data.dimensions());

    ref.setConstant(1.f);
    data.setZero();
    data(N / 2) = sqrt(N); // Parseval's theorem
    fft->forward(data);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-3f));
    fft->reverse(data);
    ref.setZero();
    ref(N / 2) = sqrt(N);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-3f));
  }
}
