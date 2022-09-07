#include "../src/op/nufft.hpp"
#include "../src/precond/single.hpp"
#include "log.h"
#include <catch2/catch.hpp>
using namespace rl;

TEST_CASE("NUFFT", "[nufft]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = GENERATE(7, 15, 16);
  Info const info{.channels = 1, .samples = 3, .traces = 1, .matrix = Sz3{M, M, M}};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f;
  points(1, 0, 0) = -0.4f;
  points(0, 2, 0) = 0.4f;
  points(1, 2, 0) = 0.4f;
  Trajectory const traj(info, points);

  float const osamp = GENERATE(2.f, 2.7f, 3.f);
  std::string const ktype = GENERATE("ES7");
  auto gridder = make_grid<Cx, 3>(traj, ktype, osamp, info.channels);
  NUFFTOp nufft(LastN<3>(info.matrix), gridder.get());
  Cx3 ks(nufft.outputDimensions());
  Cx5 img(nufft.inputDimensions());
  ks.setConstant(1.f);
  img = nufft.adjoint(ks);
  CHECK(Norm(img) == Approx(Norm(ks)).margin(5.e-2f));
  ks = nufft.forward(img);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(5.e-2f));
}

TEST_CASE("Preconditioner", "[precond]")
{
  // Log::SetLevel(Log::Level::Debug);
  Index const M = 7;
  Info const info{.channels = 1, .samples = 3, .traces = 1, .matrix = Sz3{M, M, M}};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.25f;
  points(0, 2, 0) = 0.25f;
  Trajectory const traj(info, points);
  SingleChannel sc(traj);
  CHECK(sc.pre_(0, 0, 0) == Approx(1.f).margin(1.e-6f));
  CHECK(sc.pre_(0, 1, 0) == Approx(1.f).margin(1.e-6f));
  CHECK(sc.pre_(0, 2, 0) == Approx(1.f).margin(1.e-6f));
}
