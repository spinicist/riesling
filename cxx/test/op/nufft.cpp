#include "rl/op/nufft.hpp"
#include "rl/log/log.hpp"
#include "rl/op/grid.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("NUFFT", "[op]")
{
  // Log::SetDisplayLevel(Log::Display::High);
  Index const M = GENERATE(6);
  auto const  matrix = Sz1{M};
  Re3         points(1, M, 1);
  points.setZero();
  for (Index ii = 0; ii < M; ii++) {
    points(0, ii, 0) = -0.5f * M + ii;
  }
  TrajectoryN<1> const traj(points, matrix);
  Basis                basis;
  float const          osamp = GENERATE(2.f, 2.3f);
  auto                 nufft = TOps::NUFFT<1>(GridOpts<1>{.osamp = osamp}, traj, 1, &basis);
  Cx3                  ks(nufft.oshape);
  Cx3                  img(nufft.ishape);
  img.setZero();
  img(M / 2, 0, 0) = std::sqrt(M);
  ks = nufft.forward(img);
  INFO("OSAMP " << osamp);
  INFO("IMG\n" << img);
  INFO("KS\n" << ks);
  CHECK(Norm<false>(ks) == Approx(Norm<false>(img)).margin(1.e-2f));
  Cx3 img2 = nufft.adjoint(ks);
  INFO("IMG2\n" << img2);
  CHECK(Norm<false>(img2) == Approx(Norm<false>(ks)).margin(1.e-2f));
  CHECK(Norm<false>(img - img2) == Approx(0).margin(1.e-2f));
}

TEST_CASE("NUFFTB", "[op]")
{
  Index const M = 6;
  auto const  matrix = Sz1{M};
  Index const N = 8;
  Re3         points(1, 1, N);
  points.setZero();
  TrajectoryN<1> const traj(points, matrix);

  Index const O = 4;
  Basis       basis(O, 1, N);
  basis.B.setZero();
  Index const P = N / O;
  for (Index ii = 0; ii < O; ii++) {
    for (Index ij = 0; ij < P; ij++) {
      basis.B(ii, 0, (ii * P) + ij) = std::pow(-1.f, ii) / std::sqrt(P);
    }
  }

  auto nufft = TOps::NUFFT<1>(GridOpts<1>{.osamp = 2.f}, traj, 1, &basis);
  Cx3  ks(nufft.oshape);
  ks.setConstant(1.f);
  Cx3 img(nufft.ishape);
  img.setZero();
  img = nufft.adjoint(ks);
  ks = nufft.forward(img);
  CHECK(std::real(ks(0, 0, 0)) == Approx(1.f).margin(2.e-2f));
}
