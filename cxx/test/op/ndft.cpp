#include "rl/op/ndft.hpp"
#include "rl/log/log.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("NDFT", "[op]")
{
  Threads::SetGlobalThreadCount(1);
  Index const M = GENERATE(5, 6);
  Re3         points(1, M, 1);
  points.setZero();
  for (Index ii = 0; ii < M; ii++) {
    points(0, ii, 0) = -(float)M / 2 + ii;
  }
  TOps::NDFT<1> ndft(Sz1{M}, points, 1, nullptr);
  Cx3           ks(ndft.oshape);
  Cx3           img(ndft.ishape);
  img.setZero();
  img(M / 2, 0, 0) = std::sqrt(M);
  ks = ndft.forward(img);
  INFO("IMG\n" << img);
  INFO("KS\n" << ks);
  CHECK(Norm<false>(ks) == Approx(Norm<false>(img)).margin(1.e-6f));
  for (Index ii = 0; ii < M; ii++) {
    CHECK(ks(0, ii, 0).real() == Approx(1.0).margin(1.e-6f));
    CHECK(ks(0, ii, 0).imag() == Approx(0.0).margin(1.e-6f));
  }
  Cx3 img2 = ndft.adjoint(ks);
  INFO("IMG2\n" << img2);
  CHECK(Norm<false>(img) == Approx(Norm<false>(ks)).margin(1.e-6f));
  CHECK(Norm<false>(img - img2) == Approx(0.0).margin(1.e-6f));
}

TEST_CASE("NDFT-Basis", "[op]")
{
  Index const M = 8;
  Index const N = 5;
  Re3         points(1, 1, M);
  points.setZero();
  Index const O = 4;
  Basis       basis(O, 1, M);
  basis.B.setZero();
  Index const P = M / O;
  for (Index ii = 0; ii < O; ii++) {
    for (Index ij = 0; ij < P; ij++) {
      basis.B(ii, 0, (ii * P) + ij) = std::pow(-1.f, ii) / std::sqrt(P);
    }
  }
  TOps::NDFT<1> ndft(Sz1{N}, points, 1, &basis);
  Cx3           ks(ndft.oshape);
  ks.setConstant(1.f);
  Cx3 img(ndft.ishape);
  img.setZero();
  img = ndft.adjoint(ks);
  ks = ndft.forward(img);
  INFO("BASIS\n" << basis.B);
  INFO("IMG\n" << img);
  INFO("KS\n" << ks);
  CHECK(std::real(ks(0, 0, 0)) == Approx(1.f).margin(2.e-2f));
}
