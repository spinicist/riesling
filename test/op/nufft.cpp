#include "op/nufft.hpp"
#include "kernel/expsemi.hpp"
#include "kernel/rectilinear.hpp"
#include "log.hpp"
#include "op/grid.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("NUFFT", "[nufft]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = 5;
  Info const info{.matrix = Sz3{M, M, M}};
  Re3 points(1, 5, 1);
  points.setZero();
  points(0, 0, 0) = -0.5f;
  points(0, 1, 0) = -0.25f;
  points(0, 2, 0) = 0.f;
  points(0, 3, 0) = 0.25f;
  points(0, 4, 0) = 0.5f;
  Trajectory const traj(info, points);

  float const osamp = GENERATE(2.f, 2.3f);
  using Kernel = Rectilinear<1, ExpSemi<3>>;
  Mapping<1> mapping(traj, osamp, Kernel::PadWidth);

  std::shared_ptr<GridBase<Cx, 1>> grid = std::make_shared<Grid<Cx, Kernel>>(mapping, 1);
  Index const N = 5;
  NUFFTOp<1> nufft(grid, Sz1{N});
  Cx3 ks(nufft.outputDimensions());
  Cx3 img(nufft.inputDimensions());
  img.setZero();
  img(0, 0, N / 2) = std::sqrt(N);
  ks = nufft.forward(img);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(2.e-2f));
  img = nufft.adjoint(ks);
  CHECK(Norm(img) == Approx(Norm(ks)).margin(2.e-2f));
}
