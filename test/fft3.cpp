#include "../src/fft/fft.hpp"
#include "../src/log.hpp"
#include "../src/tensorOps.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/catch_approx.hpp>
#include <fmt/format.h>

using namespace rl;
using namespace Catch;

TEST_CASE("FFT3","[FFT]")
{
  auto sx = GENERATE(3, 5, 7, 32);
  auto sy = 1;
  SECTION("<3, 3>")
  {
    auto sz = GENERATE(1, 2, 3, 16, 32);
    INFO("FFT shape: " << sx << "," << sy << "," << sz);
    Index const N = sx * sy * sz;
    Cx3 data(sx, sy, sz);
    Cx3 ref(sx, sy, sz);
    auto const fft = FFT::Make<3, 3>(data.dimensions());

    ref.setConstant(1.f);
    data.setZero();
    data(sx / 2, sy / 2, sz / 2) = sqrt(N); // Parseval's theorem
    fft->forward(data);
    INFO("data\n" << data << "\nref\n" << ref);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-3f));
    fft->reverse(data);
    ref.setZero();
    ref(sx / 2, sy / 2, sz / 2) = sqrt(N);
    INFO("data\n" << data << "\nref\n" << ref);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-3f));
  }

  SECTION("<5, 3>")
  {
    auto sz = GENERATE(1, 3, 7, 8, 16);
    Index const N = sx * sy * sz;
    Index const nc = 4;
    Index const ne = 1;
    Cx5 data(nc, ne, sx, sy, sz);
    Cx5 ref(nc, ne, sx, sy, sz);
    auto const fft = FFT::Make<5, 3>(data.dimensions());

    ref.setConstant(1.f);
    data.setZero();
    data.chip(sz / 2, 4).chip(sy / 2, 3).chip(sx / 2, 2).setConstant(sqrt(N));
    fft->forward(data);
    INFO("data\n" << data << "\nref\n" << ref);
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-6f * N * nc));
    fft->reverse(data);
    ref.setZero();
    ref.chip(sz / 2, 4).chip(sy / 2, 3).chip(sx / 2, 2).setConstant(sqrt(N));
    CHECK(Norm(data - ref) == Approx(0.f).margin(1.e-6f * N * nc));
  }
}
