#include "../src/precond/single.hpp"
#include "log.h"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Preconditioner", "[precond]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = GENERATE(31, 32);
  Info const info{.matrix = Sz3{M, M, M}, .channels = 1, .samples = 3, .traces = 1};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.25f;
  points(0, 2, 0) = 0.25f;
  Trajectory const traj(info, points);
  SingleChannel sc(traj);
  CHECK(sc.pre_(0, 0, 0) == Approx(1.f).margin(1.e-1f));
  CHECK(sc.pre_(0, 1, 0) == Approx(1.f).margin(1.e-1f));
  CHECK(sc.pre_(0, 2, 0) == Approx(1.f).margin(1.e-1f));
}
