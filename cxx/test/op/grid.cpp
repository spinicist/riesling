#include "rl/op/grid.hpp"
#include "rl/log.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <numbers>

using namespace rl;
using namespace Catch;

constexpr float inv_sqrt2 = 1.f / std::numbers::sqrt2;

TEST_CASE("Grid-Basic", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  Threads::SetGlobalThreadCount(1);
  Index const M = GENERATE(16, 32);
  auto const  matrix = Sz2{M, M};
  Re3         points(2, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f * M;
  points(1, 0, 0) = -0.4f * M;
  points(0, 2, 0) = 0.4f * M;
  points(1, 2, 0) = 0.4f * M;
  TrajectoryN<2> const traj(points, matrix);
  Basis                basis;

  float const osamp = GENERATE(1.3f, 2.f);
  auto        grid = TOps::Grid<2>::Make(TOps::Grid<2>::Opts{.osamp = osamp, .ktype = "ES6"}, traj, 1, &basis);
  Cx3         noncart(grid->oshape);
  Cx4         cart(grid->ishape);
  noncart.setConstant(1.f);
  cart = grid->adjoint(noncart);
  INFO("M " << M << " OS " << osamp);
  INFO("noncart\n" << noncart);
  INFO("cart\n" << cart);
  auto const cs = std::sqrt(cart.size());
  CHECK((Norm<false>(cart) - Norm<false>(noncart)) / cs == Approx(0.f).margin(2e-4f));
  noncart = grid->forward(cart);
  INFO("noncart\n" << noncart);
  CHECK((Norm<false>(cart) - Norm<false>(noncart)) / cs == Approx(0.f).margin(2e-4f));
}

TEST_CASE("Grid-Basis-Sample", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  Threads::SetGlobalThreadCount(1);
  Index const M = 6;
  auto const  matrix = Sz1{M};
  Re3         points(1, 6, 1);
  points.setZero();
  points(0, 0, 0) = -3.f;
  points(0, 1, 0) = -2.f;
  points(0, 2, 0) = -1.f;
  points(0, 3, 0) = 0;
  points(0, 4, 0) = 1.f;
  points(0, 5, 0) = 2.f;
  TrajectoryN<1> const traj(points, matrix);

  float const osamp = 1;
  Basis       basis(2, 6, 1);
  basis.B.setZero();
  basis.B(0, 0, 0) = 1.f;
  basis.B(0, 1, 0) = 1.f;
  basis.B(0, 2, 0) = 1.f;
  basis.B(1, 3, 0) = 1.f;
  basis.B(1, 4, 0) = 1.f;
  basis.B(1, 5, 0) = 1.f;
  auto grid = TOps::Grid<1>::Make(TOps::Grid<1>::Opts{.osamp = 1.f, .ktype = "NN"}, traj, 1, &basis);
  Cx3  noncart(grid->oshape);
  noncart.setConstant(1.f);
  Cx3 cart = grid->adjoint(noncart);
  INFO("GRID\n" << cart);
  CHECK(cart(0, 0, 0).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(1, 0, 0).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(2, 0, 0).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(3, 0, 0).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(4, 0, 0).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(5, 0, 0).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(0, 0, 1).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(1, 0, 1).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(2, 0, 1).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(3, 0, 1).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(4, 0, 1).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(5, 0, 1).real() == Approx(1.f).margin(1e-2f));
  Cx3 nc2 = grid->forward(cart);
  INFO("NC\n" << nc2);
  CHECK(Norm<false>(nc2 - noncart) == Approx(0.f).margin(1e-2f));
}

TEST_CASE("GridVCC", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  Threads::SetGlobalThreadCount(0);
  Index const M = 8;
  auto const  matrix = Sz1{M};
  Re3         points(1, M / 2, 1);
  for (Index ii = 0; ii < M / 2; ii++) {
    points(0, ii, 0) = ii;
  }
  TrajectoryN<1> const traj(points, matrix);
  Basis                basis;

  auto grid = TOps::Grid<1>::Make(TOps::Grid<1>::Opts{.osamp = 1.f, .ktype = "NN", .vcc = true}, traj, 1, &basis);
  Cx3  noncart(grid->oshape);
  Cx3  cart(grid->ishape);
  noncart.setConstant(Cx(0.f, 1.f));
  cart = grid->adjoint(noncart);
  INFO("M " << M << " cart" << '\n' << cart);
  CHECK(Norm<false>(cart) == Approx(Norm<false>(noncart)).margin(1e-2f));
  CHECK(cart(M / 2, 0, 0).real() == Approx(0.f).margin(1e-6f));
  CHECK(cart(M / 2, 0, 0).imag() == Approx(inv_sqrt2).margin(1e-6f));
  CHECK(cart(M / 2, 1, 0).real() == Approx(0.f).margin(1e-6f));
  CHECK(cart(M / 2, 1, 0).imag() == Approx(-inv_sqrt2).margin(1e-6f));
  for (Index ii = 1; ii < M / 2; ii++) {
    CHECK(cart(M / 2 - ii, 0, 0).real() == Approx(0.f).margin(1e-6f));
    CHECK(cart(M / 2 - ii, 0, 0).imag() == Approx(0.f).margin(1e-6f));
    CHECK(cart(M / 2 + ii, 0, 0).real() == Approx(0.f).margin(1e-6f));
    CHECK(cart(M / 2 + ii, 0, 0).imag() == Approx(inv_sqrt2).margin(1e-6f));
    CHECK(cart(M / 2 - ii, 1, 0).real() == Approx(0.f).margin(1e-6f));
    CHECK(cart(M / 2 - ii, 1, 0).imag() == Approx(-inv_sqrt2).margin(1e-6f));
    CHECK(cart(M / 2 + ii, 1, 0).real() == Approx(0.f).margin(1e-6f));
    CHECK(cart(M / 2 + ii, 1, 0).imag() == Approx(0.f).margin(1e-6f));
  }
  noncart = grid->forward(cart);
  INFO("noncart" << '\n' << noncart);
  CHECK(Norm<false>(noncart) == Approx(Norm<false>(cart)).margin(1e-2f));
  for (Index ii = 0; ii < M / 2; ii++) {
    CHECK(noncart(ii, 0, 0).real() == Approx(0.f).margin(1e-6f));
    CHECK(noncart(ii, 0, 0).imag() == Approx(1.f).margin(1e-6f));
  }
}
