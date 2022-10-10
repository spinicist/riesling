#include "../src/op/nufft.hpp"
#include "../src/func/pre-kspace.hpp"
#include "log.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("NUFFT", "[nufft]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = GENERATE(7, 15, 16);
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
  NUFFTOp<3> nufft(traj, ktype, osamp, 1);
  Cx3 ks(nufft.outputDimensions());
  Cx5 img(nufft.inputDimensions());
  ks.setConstant(1.f);
  img = nufft.adjoint(ks);
  CHECK(Norm(img) == Approx(Norm(ks)).margin(5.e-2f));
  ks = nufft.forward(img);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(5.e-2f));
}
