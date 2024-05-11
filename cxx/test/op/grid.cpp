#include "op/grid.hpp"
#include "log.hpp"
#include "tensors.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Grid", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  Threads::SetGlobalThreadCount(1);
  Index const M = GENERATE(7, 15, 16, 31, 32);
  auto const matrix = Sz2{M, M};
  Re3 points(2, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f * M;
  points(1, 0, 0) = -0.4f * M;
  points(0, 2, 0) = 0.4f * M;
  points(1, 2, 0) = 0.4f * M;
  TrajectoryN<2> const traj(points, matrix);

  float const osamp = GENERATE(2.f, 2.7f, 3.f);
  std::string const ktype = GENERATE("ES7");
  auto basis = IdBasis<float>();
  auto grid = Grid<float, 2>::Make(traj, ktype, osamp, 1, basis);
  Re3 noncart(grid->oshape);
  Re4 cart(grid->ishape);
  noncart.setConstant(1.f);
  cart = grid->adjoint(noncart);
  INFO("M " << M << " OS " << osamp << " " << ktype);
  CHECK(Norm(cart) == Approx(Norm(noncart)).margin(1e-2f));
  noncart = grid->forward(cart);
  CHECK(Norm(noncart) == Approx(Norm(cart)).margin(1e-2f));
}

TEST_CASE("Grid Sample Basis", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  Threads::SetGlobalThreadCount(1);
  Index const M = 6;
  auto const matrix = Sz1{M};
  Re3 points(1, 6, 1);
  points.setZero();
  points(0, 0, 0) = -3.f;
  points(0, 1, 0) = -2.f;
  points(0, 2, 0) = -1.f;
  points(0, 3, 0) = 0;
  points(0, 4, 0) = 1.f;
  points(0, 5, 0) = 2.f;
  TrajectoryN<1> const traj(points, matrix);

  float const osamp = 1;
  std::string const ktype = GENERATE("NN");
  Re3 basis(2, 6, 1);
  basis.setZero();
  basis(0, 0, 0) = 1.f;
  basis(0, 1, 0) = 1.f;
  basis(0, 2, 0) = 1.f;
  basis(1, 3, 0) = 1.f;
  basis(1, 4, 0) = 1.f;
  basis(1, 5, 0) = 1.f;
  auto grid = Grid<float, 1>::Make(traj, ktype, osamp, 1, basis);
  Re3 noncart(grid->oshape);
  noncart.setConstant(1.f);
  Re3 cart = grid->adjoint(noncart);
  INFO("GRID\n" << cart);
  CHECK(cart(0, 0, 0)== Approx(1.f).margin(1e-2f));
  CHECK(cart(0, 0, 1)== Approx(1.f).margin(1e-2f));
  CHECK(cart(0, 0, 2)== Approx(1.f).margin(1e-2f));
  CHECK(cart(0, 0, 3)== Approx(0.f).margin(1e-2f));
  CHECK(cart(0, 0, 4)== Approx(0.f).margin(1e-2f));
  CHECK(cart(0, 0, 5)== Approx(0.f).margin(1e-2f));
  CHECK(cart(0, 1, 0)== Approx(0.f).margin(1e-2f));
  CHECK(cart(0, 1, 1)== Approx(0.f).margin(1e-2f));
  CHECK(cart(0, 1, 2)== Approx(0.f).margin(1e-2f));
  CHECK(cart(0, 1, 3)== Approx(1.f).margin(1e-2f));
  CHECK(cart(0, 1, 4)== Approx(1.f).margin(1e-2f));
  CHECK(cart(0, 1, 5)== Approx(1.f).margin(1e-2f));
  Re3 nc2 = grid->forward(cart);
  INFO("NC\n" << nc2);
  CHECK(Norm(nc2 - noncart)== Approx(0.f).margin(1e-2f));
}
