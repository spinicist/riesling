#include "op/nufft.hpp"
#include "basis/fourier.hpp"
#include "log.hpp"
#include "op/grid.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("NUFFT", "[tform]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = GENERATE(5, 6);
  auto const  matrix = Sz3{M, M, M};
  Re3         points(1, M, 1);
  points.setZero();
  for (Index ii = 0; ii < M; ii++) {
    points(0, ii, 0) = -0.5f * M + ii;
  }
  Trajectory const traj(points, matrix);
  float const      osamp = GENERATE(2.f, 2.3f);
  auto             grid = Grid<Cx, 1>::Make(traj, "ES3", osamp, 1);
  NUFFTOp<1>       nufft(grid, Sz1{M});
  Cx3              ks(nufft.oshape);
  Cx3              img(nufft.ishape);
  img.setZero();
  img(0, 0, M / 2) = std::sqrt(M);
  ks = nufft.forward(img);
  INFO("IMG\n" << img);
  INFO("KS\n" << ks);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(1.e-2f));
  img = nufft.adjoint(ks);
  INFO("IMG\n" << img);
  CHECK(Norm(img) == Approx(Norm(ks)).margin(1.e-2f));
}

TEST_CASE("NUFFT Basis Trace", "[tform]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = 5;
  auto const  matrix = Sz3{M, 1, 1};
  Index const N = 8;
  Re3         points(1, 1, N);
  points.setZero();
  Trajectory const traj(points, matrix);

  Index const O = 4;
  Basis<Cx>   basis(O, 1, N);
  basis.setZero();
  Index const P = N / O;
  for (Index ii = 0; ii < O; ii++) {
    for (Index ij = 0; ij < P; ij++) {
      basis(ii, 0, (ii * P) + ij) = std::pow(-1.f, ii) / std::sqrt(P);
    }
  }

  float const osamp = 2.f;
  auto        grid = Grid<Cx, 1>::Make(traj, "ES3", osamp, 1, basis);

  NUFFTOp<1> nufft(grid, Sz1{M});
  Cx3        ks(nufft.oshape);
  ks.setConstant(1.f);
  Cx3 img(nufft.ishape);
  img.setZero();
  img = nufft.adjoint(ks);
  ks = nufft.forward(img);
  CHECK(std::real(ks(0, 0, 0)) == Approx(1.f).margin(2.e-2f));
}

TEST_CASE("NUFFT Basis Fourier", "[tform]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = 8;
  auto const  matrix = Sz3{M, 1, 1};
  Re3         points(1, M, 1);
  points.setZero();
  for (Index ii = 0; ii < M; ii++) {
    points(0, ii, 0) = -0.5f * M + ii;
  }
  Trajectory const traj(points, matrix);
  Index const      N = 3;
  auto             b = FourierBasis(N, M, 1, 1.f);

  float const osamp = 2.f;
  auto        grid = Grid<Cx, 1>::Make(traj, "ES3", osamp, 1, b.basis);

  NUFFTOp<1> nufft(grid, Sz1{M});
  Cx3        ks(nufft.oshape);
  ks.setConstant(1.f);
  Cx3 img(nufft.ishape);
  img.setZero();
  img = nufft.adjoint(ks);
  ks = nufft.forward(img);
  CHECK(std::real(ks(0, 0, 0)) == Approx(1.f).margin(2.e-2f));
}
