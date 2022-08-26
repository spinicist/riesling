#include "../../src/op/gridBase.hpp"
#include "../../src/sdc.h"
#include "../../src/tensorOps.h"
#include "../../src/traj_spirals.h"
#include "../../src/trajectory.h"

#include <catch2/catch.hpp>

using namespace rl;

TEST_CASE("Grid Basic", "[grid]")
{
  // Log::SetLevel(Log::Level::Debug);
  SECTION("Grid Size")
  {
    Index const M = GENERATE(7, 15, 31);
    Info const info{.channels = 1, .samples = 3, .traces = 1, .matrix = Sz3{M, M, M}};
    Re3 points(3, 3, 1);
    points.setZero();
    points(0, 0, 0) = -0.4f;
    points(0, 2, 0) = 0.4f;
    Trajectory const traj(info, points);

    float const osamp = GENERATE(2.f, 2.7f, 3.f);
    auto const kernel = rl::make_kernel("FI5", info.grid3D, osamp);
    Mapping const mapping(traj, kernel.get(), osamp);
    auto grid = make_grid<Cx>(kernel.get(), mapping, info.channels);
    Cx3 ks(grid->outputDimensions());
    Cx5 img(grid->inputDimensions());
    ks.setConstant(1.f);
    img = grid->Adj(ks);
    CHECK(Norm(img) == Approx(std::sqrt(ks.size())).margin(5.e-2f));
    ks = grid->A(img);
    CHECK(Norm(img) == Approx(std::sqrt(ks.size())).margin(5.e-2f));
  }
}
