#include "rl/op/wavelets.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Wavelets", "[op]")
{
  Index const sz = 64;

  SECTION("1D")
  {
    Sz4 const          shape{sz, 1, 1, 1};
    std::vector<Index> dims{0};
    TOps::Wavelets     wave(shape, 6, dims);
    Cx4                x(shape);
    x.setZero();
    x(sz / 2, 0, 0, 0) = Cx(1.f);

    Cx4 y = wave.forward(x);
    Cx4 xx = wave.adjoint(y);

    INFO("y\n" << y);
    INFO("xx\n" << xx);
    CHECK(Norm<false>(x - xx) == Approx(0).margin(1.e-6f));
  }

  SECTION("2D")
  {
    Sz4 const          shape{sz, sz, 1, 1};
    std::vector<Index> dims{0, 1};
    TOps::Wavelets     wave(shape, 6, dims);
    Cx4                x(shape);
    x.setZero();
    x(sz / 2, sz / 2, 0, 0) = Cx(1.f);

    Cx4 y = wave.forward(x);
    Cx4 xx = wave.adjoint(y);

    INFO("y\n" << y);
    INFO("xx\n" << xx);
    CHECK(Norm<false>(x - xx) == Approx(0).margin(1.e-6f));
  }

  SECTION("3D")
  {
    Sz4 const          shape{1, sz, sz, sz};
    std::vector<Index> dims{1, 2, 3};
    TOps::Wavelets     wave(shape, 6, dims);
    Cx4                x(shape);
    x.setZero();
    x(0, sz / 2, sz / 2, sz / 2) = Cx(1.f);

    Cx4 y = wave.forward(x);
    Cx4 xx = wave.adjoint(y);

    INFO("y\n" << y);
    INFO("xx\n" << xx);
    CHECK(Norm<false>(x - xx) == Approx(0).margin(1.e-6f));
  }
}