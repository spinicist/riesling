#include "../../src/op/gridBase.hpp"
#include "log.h"
#include "tensorOps.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Grid Basic", "[grid]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = GENERATE(7, 15, 16, 31, 32);
  Info const info{.matrix = Sz3{M, M, 1}, .channels = 1, .samples = 3, .traces = 1};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f;
  points(1, 0, 0) = -0.4f;
  points(0, 2, 0) = 0.4f;
  points(1, 2, 0) = 0.4f;
  Trajectory const traj(info, points);

  float const osamp = GENERATE(2.f, 2.7f, 3.f);
  std::string const ktype = GENERATE("ES7");
  auto grid = make_grid<float, 2>(traj, ktype, osamp, info.channels);
  Re3 ks(grid->outputDimensions());
  Re4 img(grid->inputDimensions());
  ks.setConstant(1.f);
  img = grid->adjoint(ks);
  INFO("M " << M << " OS " << osamp << " " << ktype);
  INFO(img.chip(0, 1).chip(0, 0));
  CHECK(Norm(img) == Approx(Norm(ks)).margin(1e-2f));
  ks = grid->forward(img);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(1e-2f));
}
