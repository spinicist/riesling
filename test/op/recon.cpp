#include "../src/op/recon.hpp"
#include "log.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Recon", "[recon]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = GENERATE(7, 15, 16);
  Index const nC = 4;
  Info const info{.matrix = Sz3{M, M, M}};
  Re3 points(3, 3, 1);
  points.setZero();
  points(0, 0, 0) = -0.4f;
  points(1, 0, 0) = -0.4f;
  points(0, 2, 0) = 0.4f;
  points(1, 2, 0) = 0.4f;
  Trajectory const traj(info, points);

  float const osamp = GENERATE(2.f, 2.7f, 3.f);
  std::string const ktype = GENERATE("ES7");
  auto grid = make_grid<Cx, 3>(traj, ktype, osamp, nC);
  auto nufft = make_nufft(traj, ktype, osamp, nC, traj.matrix());

  Cx4 senseMaps(AddFront(traj.matrix(), nC));
  senseMaps.setConstant(std::sqrt(0.25f));
  auto sense = std::make_shared<SenseOp>(senseMaps, traj.nFrames());
  MultiplyOp<SenseOp, Operator<Cx, 5, 4>> recon("ReconOp", sense, nufft);

  Cx4 ks(recon.outputDimensions());
  Cx4 img(recon.inputDimensions());
  ks.setConstant(1.f);
  img = recon.adjoint(ks);
  // Super loose tolerance
  CHECK(Norm(img) == Approx(Norm(ks)).margin(2.e-1f));
  ks = recon.forward(img);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(2.e-1f));
}
