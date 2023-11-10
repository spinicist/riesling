#include "../src/precond.hpp"
#include "../src/basis.hpp"
#include "log.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Preconditioner", "[precond]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = GENERATE(15, 16);
  Info const info{.matrix = Sz3{M, M, M}};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.25f;
  points(0, 2, 0) = 0.25f;
  Trajectory const traj(info, points);
  Re2 basis(1, 1);
  basis.setConstant(1.f);
  Re2 sc = KSpaceSingle(traj, IdBasis(), 0.f);
  CHECK(sc(0, 0) == Approx(1.f).margin(1.e-1f));
  CHECK(sc(1, 0) == Approx(1.f).margin(1.e-1f));
  CHECK(sc(2, 0) == Approx(1.f).margin(1.e-1f));
}
