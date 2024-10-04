#include "op/nufft.hpp"
#include "basis/fourier.hpp"
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
  auto                 nufft = TOps::NUFFT<1, false>::Make(traj, "ES3", osamp, 1, &basis, traj.matrix());
  Cx3                  ks(nufft->oshape);
  Cx3                  img(nufft->ishape);
  img.setZero();
  img(0, 0, M / 2) = std::sqrt(M);
  ks = nufft->forward(img);
  INFO("IMG\n" << img);
  INFO("KS\n" << ks);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(1.e-2f));
  img = nufft->adjoint(ks);
  INFO("IMG\n" << img);
  CHECK(Norm(img) == Approx(Norm(ks)).margin(1.e-2f));
}

TEST_CASE("NUFFT-Basis", "[nufft]")
{
  Log::SetLevel(Log::Level::Testing);
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

  float const osamp = 2.f;
  auto        nufft = TOps::NUFFT<1>::Make(traj, "ES3", osamp, 1, &basis, traj.matrix());
  Cx3         ks(nufft->oshape);
  ks.setConstant(1.f);
  Cx3 img(nufft->ishape);
  img.setZero();
  img = nufft->adjoint(ks);
  ks = nufft->forward(img);
  CHECK(std::real(ks(0, 0, 0)) == Approx(1.f).margin(2.e-2f));
}

TEST_CASE("NUFFT-Batch", "[nufft]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const C = 8;
  Index const M = 8;
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

  float const osamp = 2.f;
  auto        nufft = TOps::NUFFT<1>::Make(traj, "ES3", osamp, C, &basis);
  Cx3         ks(nufft->oshape);
  ks.setConstant(1.f);
  Cx3 img(nufft->ishape);
  img.setZero();
  img = nufft->adjoint(ks);
  ks = nufft->forward(img);
  CHECK(std::real(ks(0, 0, 0)) == Approx(1.f).margin(2.e-2f));
}

TEST_CASE("NUFFT-VCC", "[nufft]")
{
  Index const M = GENERATE(8, 10);
  auto const  matrix = Sz1{M};
  Re3         points(1, 1, 1);
  points.setZero();

  TrajectoryN<1> const traj(points, matrix);
  Basis                basis;
  auto                 nufft = TOps::NUFFT<1, true>::Make(traj, "NN", 1.f, 1, &basis);
  Cx3                  ks(nufft->oshape);
  // Purely imaginary, odd symmetric
  ks.setConstant(Cx(0.f, 1.f));
  Cx4 img = nufft->adjoint(ks);
  INFO("IMG\n" << img);
  INFO("dims " << img.dimensions());
  CHECK(Norm(img) == Approx(1.f).margin(1.e-2f));
  for (Index ii = 0; ii < M; ii++) {
    CHECK(img(0, 0, 0, ii).real() == Approx(-img(0, 0, 1, ii).real()).margin(1e-6f));
    CHECK(img(0, 0, 0, ii).imag() == Approx(-img(0, 0, 1, ii).imag()).margin(1e-6f));
  }
  ks = nufft->forward(img);
  INFO("KS\n" << ks);
  CHECK(Norm(ks) == Approx(1.f).margin(1.e-2f));
}
