#include "op/grid.hpp"
#include "log.hpp"
#include "tensors.hpp"
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

  float const       osamp = GENERATE(1.3f, 2.f);
  std::string const ktype = "ES6";
  auto              grid = TOps::Grid<2, false>::Make(traj, traj.matrix(), osamp, ktype, 1, &basis);
  Cx3               noncart(grid->oshape);
  Cx4               cart(grid->ishape);
  noncart.setConstant(1.f);
  cart = grid->adjoint(noncart);
  INFO("M " << M << " OS " << osamp << " " << ktype);
  INFO("noncart\n" << noncart);
  INFO("cart\n" << cart);
  auto const cs = std::sqrt(cart.size());
  CHECK((Norm(cart) - Norm(noncart)) / cs == Approx(0.f).margin(2e-4f));
  noncart = grid->forward(cart);
  INFO("noncart\n" << noncart);
  CHECK((Norm(cart) - Norm(noncart)) / cs == Approx(0.f).margin(2e-4f));
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

  float const       osamp = 1;
  std::string const ktype = GENERATE("NN");
  Basis             basis(2, 6, 1);
  basis.B.setZero();
  basis.B(0, 0, 0) = 1.f;
  basis.B(0, 1, 0) = 1.f;
  basis.B(0, 2, 0) = 1.f;
  basis.B(1, 3, 0) = 1.f;
  basis.B(1, 4, 0) = 1.f;
  basis.B(1, 5, 0) = 1.f;
  auto grid = TOps::Grid<1, false>::Make(traj, traj.matrix(), osamp, ktype, 1, &basis);
  Cx3  noncart(grid->oshape);
  noncart.setConstant(1.f);
  Cx3 cart = grid->adjoint(noncart);
  INFO("GRID\n" << cart);
  CHECK(cart(0, 0, 0).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(0, 0, 1).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(0, 0, 2).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(0, 0, 3).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(0, 0, 4).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(0, 0, 5).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(1, 0, 0).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(1, 0, 1).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(1, 0, 2).real() == Approx(0.f).margin(1e-2f));
  CHECK(cart(1, 0, 3).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(1, 0, 4).real() == Approx(1.f).margin(1e-2f));
  CHECK(cart(1, 0, 5).real() == Approx(1.f).margin(1e-2f));
  Cx3 nc2 = grid->forward(cart);
  INFO("NC\n" << nc2);
  CHECK(Norm(nc2 - noncart) == Approx(0.f).margin(1e-2f));
}

TEST_CASE("Grid VCC", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  Threads::SetGlobalThreadCount(0);
  Index const M = GENERATE(16);
  auto const  matrix = Sz1{M};
  Re3         points(1, M / 2, 1);
  for (Index ii = 0; ii < M / 2; ii++) {
    points(0, ii, 0) = ii;
  }
  TrajectoryN<1> const traj(points, matrix);
  Basis                basis;

  auto grid = TOps::Grid<1, true>::Make(traj, traj.matrix(), 1.f, "NN", 1, &basis, true);
  Cx3  noncart(grid->oshape);
  Cx4  cart(grid->ishape);
  noncart.setConstant(Cx(0.f, 1.f));
  cart = grid->adjoint(noncart);
  INFO("M " << M << " cart" << '\n' << cart);
  CHECK(Norm(cart) == Approx(Norm(noncart)).margin(1e-2f));
  CHECK(cart(0, 0, 0, M / 2).real() == Approx(0.f).margin(1e-6f));
  CHECK(cart(0, 0, 0, M / 2).imag() == Approx(inv_sqrt2).margin(1e-6f));
  CHECK(cart(0, 0, 1, M / 2).real() == Approx(0.f).margin(1e-6f));
  CHECK(cart(0, 0, 1, M / 2).imag() == Approx(-inv_sqrt2).margin(1e-6f));
  for (Index ii = 1; ii < M / 2; ii++) {
    CHECK(cart(0, 0, 0, M / 2 - ii).real() == Approx(0.f).margin(1e-6f));
    CHECK(cart(0, 0, 0, M / 2 - ii).imag() == Approx(0.f).margin(1e-6f));
    CHECK(cart(0, 0, 0, M / 2 + ii).real() == Approx(0.f).margin(1e-6f));
    CHECK(cart(0, 0, 0, M / 2 + ii).imag() == Approx(inv_sqrt2).margin(1e-6f));
    CHECK(cart(0, 0, 1, M / 2 - ii).real() == Approx(0.f).margin(1e-6f));
    CHECK(cart(0, 0, 1, M / 2 - ii).imag() == Approx(-inv_sqrt2).margin(1e-6f));
    CHECK(cart(0, 0, 1, M / 2 + ii).real() == Approx(0.f).margin(1e-6f));
    CHECK(cart(0, 0, 1, M / 2 + ii).imag() == Approx(0.f).margin(1e-6f));
  }
  noncart = grid->forward(cart);
  INFO("noncart" << '\n' << noncart);
  CHECK(Norm(noncart) == Approx(Norm(cart)).margin(1e-2f));
  for (Index ii = 0; ii < M / 2; ii++) {
    CHECK(noncart(0, ii, 0).real() == Approx(0.f).margin(1e-6f));
    CHECK(noncart(0, ii, 0).imag() == Approx(1.f).margin(1e-6f));
  }
}
