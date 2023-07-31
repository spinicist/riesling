#include "../../src/op/grid.hpp"
#include "log.hpp"
#include "tensorOps.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Grid Basic", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  Threads::SetGlobalThreadCount(1);
  Index const M = GENERATE(7, 15, 16, 31, 32);
  Info const info{.matrix = Sz3{M, M, 1}};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f;
  points(1, 0, 0) = -0.4f;
  points(0, 2, 0) = 0.4f;
  points(1, 2, 0) = 0.4f;
  Trajectory const traj(info, points);

  float const osamp = GENERATE(2.f, 2.7f, 3.f);
  std::string const ktype = GENERATE("ES7");
  Re2 basis(1, 1);
  basis.setConstant(1.f);
  auto grid = Grid<float, 2>::Make(traj, ktype, osamp, 1, basis);
  Re3 ks(grid->oshape);
  Re4 img(grid->ishape);
  ks.setConstant(1.f);
  img = grid->adjoint(ks);
  INFO("M " << M << " OS " << osamp << " " << ktype);
  CHECK(Norm(img) == Approx(Norm(ks)).margin(1e-2f));
  ks = grid->forward(img);
  INFO(ks);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(1e-2f));
}
