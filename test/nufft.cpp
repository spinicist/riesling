#include "../src/op/nufft.hpp"
#include "../src/precond/single.hpp"
#include <catch2/catch.hpp>

#include "log.h"
using namespace rl;

TEST_CASE("NUFFT", "[nufft]")
{
  SECTION("Oversamp") {
    Log::SetLevel(Log::Level::Debug);
    Index const M = GENERATE(7,15,31);
    Info const info{
      .type = Info::Type::ThreeD, .matrix = Eigen::Array3l(M,M,M), .channels = 1, .read_points = 3, .spokes = 1};
    Re3 points(3, 3, 1);
    points.setZero();
    points(0, 0, 0) = -0.4f;
    points(0, 2, 0) = 0.4f;
    Trajectory const traj(info, points);

    float const osamp = GENERATE(2.7f);
    auto const kernel = rl::make_kernel("FI5", info.type, osamp);
    Mapping const mapping(traj, kernel.get(), osamp);
    auto gridder = make_grid<Cx>(kernel.get(), mapping, info.channels);
    NUFFTOp nufft(LastN<3>(info.matrix), gridder.get());

    Cx3 ks(nufft.outputDimensions());
    Cx5 img(nufft.inputDimensions());
    ks.setConstant(1.f);
    // ks(0,1,0) = 1.f;
    img = nufft.Adj(ks);
    CHECK(Norm(img) == Approx(std::sqrt(ks.size())).margin(5.e-2f));
    ks = nufft.A(img);
    CHECK(Norm(ks) == Approx(std::sqrt(ks.size())).margin(5.e-2f));
  }
}

TEST_CASE("Preconditioner", "[precond]")
{
  // Log::SetLevel(Log::Level::Debug);
  Index const M = 15;
  Info const info{
    .type = Info::Type::ThreeD, .matrix = Eigen::Array3l::Constant(M), .channels = 1, .read_points = 3, .spokes = 1};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.25f;
  points(0, 2, 0) = 0.25f;
  Trajectory const traj(info, points);
  SingleChannel sc(traj);

  fmt::print("sc\n{}\n", sc.pre_);
}
