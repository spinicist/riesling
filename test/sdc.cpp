#include "../src/sdc.h"
#include "../src/log.h"
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
  Info const info{.matrix = Sz3{M, M, M}, .channels = 1, .samples = 3, .traces = 1};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 1, 0) = -0.25;
  points(0, 2, 0) =  0.25;
  Trajectory const traj(info, points);

  SECTION("Pipe-NN")
  {
    Re2 sdc = SDC::Pipe(traj, "NN", 2.f);
    CHECK(sdc.dimension(0) == info.samples);
    CHECK(sdc.dimension(1) == info.traces);
    CHECK(sdc(0, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(1, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(2, 0) == Approx(1.f).margin(1.e-6f));
  }

  SECTION("Pipe")
  {
    Re2 sdc = SDC::Pipe(traj, "ES7", 2.1f);
    CHECK(sdc.dimension(0) == info.samples);
    CHECK(sdc.dimension(1) == info.traces);
    CHECK(sdc(0, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(1, 0) == Approx(1.f).margin(1.e-6f));
    CHECK(sdc(2, 0) == Approx(1.f).margin(1.e-6f));
  }
}
