#include "op/ndft.hpp"
#include "log.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("NDFT", "[ndft]")
{
  Log::SetLevel(Log::Level::Debug);
  Index const M = 5;
  Index const N = 6;
  Re3         points(1, M, 1);
  points.setZero();
  points(0, 0, 0) = -0.5f;
  points(0, 1, 0) = -0.25f;
  points(0, 2, 0) = 0.f;
  points(0, 3, 0) = 0.25f;
  points(0, 4, 0) = 0.5f;
  NDFTOp<1>   ndft(points, 1, Sz1{N});
  Cx3         ks(ndft.oshape);
  Cx3         img(ndft.ishape);
  img.setZero();
  img(0, 0, 0) = 1.f;
  img(0, 0, 1) = -1.f;
  img(0, 0, 2) = 1.f;
  img(0, 0, 3) = -1.f;
  img(0, 0, 4) = 1.f;
  img(0, 0, 5) = -1.f;

  // img(0, 0, N / 2) = std::sqrt(N);
  ks = ndft.forward(img);
  INFO("IMG\n" << img);
  INFO("KS\n" << ks);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(2.e-2f));
  img = ndft.adjoint(ks);
  INFO("IMG\n" << img);
  CHECK(Norm(img) == Approx(Norm(ks)).margin(2.e-2f));
}

TEST_CASE("NDFT Basis", "[ndft]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = 8;
  Index const N = 5;
  Re3         points(1, 1, M);
  points.setZero();
  Index const O = 4;
  Re2         basis(O, M);
  basis.setZero();
  Index const P = M / O;
  for (Index ii = 0; ii < O; ii++) {
    for (Index ij = 0; ij < P; ij++) {
      basis(ii, (ii * P) + ij) = std::pow(-1.f, ii) / std::sqrt(P);
    }
  }
  NDFTOp<1> ndft(points, 1, Sz1{N});
  Cx3       ks(ndft.oshape);
  ks.setConstant(1.f);
  Cx3 img(ndft.ishape);
  img.setZero();
  img = ndft.adjoint(ks);
  ks = ndft.forward(img);
  CHECK(std::real(ks(0, 0, 0)) == Approx(1.f).margin(2.e-2f));
}
