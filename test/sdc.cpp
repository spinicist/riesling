#include "../src/sdc.hpp"
#include "../src/log.hpp"
#include "../src/traj_spirals.h"
#include "../src/trajectory.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("SDC", "[sdc]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = 16;
  Info const info{.matrix = Sz3{M, M, M}};
  Index const samples = 3, traces = 1;
  Re3 points(3, samples, traces);
  points.setZero();
  points(0, 1, 0) = -0.25;
  points(0, 2, 0) =  0.25;
  Trajectory const traj(info, points);

  SECTION("Pipe-NN")
  {
    Re2 sdc = SDC::Pipe<3>(traj, "NN", 2.f);
    CHECK(sdc.dimension(0) == samples);
    CHECK(sdc.dimension(1) == traces);
    CHECK(sdc(0, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(1, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(2, 0) == Approx(1.f).margin(1.e-6f));
  }

  SECTION("Pipe")
  {
    Re2 sdc = SDC::Pipe<3>(traj, "ES7", 2.1f);
    CHECK(sdc.dimension(0) == samples);
    CHECK(sdc.dimension(1) == traces);
    CHECK(sdc(0, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(1, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(2, 0) == Approx(1.f).margin(1.e-6f));
  }
}
