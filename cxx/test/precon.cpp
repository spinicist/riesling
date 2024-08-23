#include "precon.hpp"
#include "basis/basis.hpp"
#include "log.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Preconditioner", "[precon]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = GENERATE(15, 16);
  Sz3 const matrix{M, M, M};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.25f * M;
  points(0, 2, 0) = 0.25f * M;
  Trajectory const traj(points, matrix);
  Basis basis;
  Re2 sc = KSpaceSingle(traj, &basis, false, 0.f);
  CHECK(sc(0, 0) == Approx(1.f).margin(1.e-1f));
  CHECK(sc(1, 0) == Approx(1.f).margin(1.e-1f));
  CHECK(sc(2, 0) == Approx(1.f).margin(1.e-1f));
}
