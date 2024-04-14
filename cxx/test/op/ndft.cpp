#include "op/ndft.hpp"
#include "log.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("NDFT", "[tform]")
{
  Log::SetLevel(Log::Level::Testing);
  Threads::SetGlobalThreadCount(1);
  Index const M = GENERATE(5, 6);
  Re3         points(1, M, 1);
  points.setZero();
  for (Index ii = 0; ii < M; ii++) {
    points(0, ii, 0) = -0.5f + ii / (float)M;
  }
  NDFTOp<1> ndft(points, 1, Sz1{M});
  Cx3       ks(ndft.oshape);
  Cx3       img(ndft.ishape);
  img.setZero();
  img(0, 0, M / 2) = std::sqrt(M);
  ks = ndft.forward(img);
  INFO("IMG\n" << img);
  INFO("KS\n" << ks);
  CHECK(Norm(ks) == Approx(Norm(img)).margin(2.e-2f));
  img = ndft.adjoint(ks);
  INFO("IMG\n" << img);
  CHECK(Norm(img) == Approx(Norm(ks)).margin(2.e-2f));
}

TEST_CASE("NDFT Basis", "[tform]")
{
  Log::SetLevel(Log::Level::Testing);
  Index const M = 8;
  Index const N = 5;
  Re3         points(1, 1, M);
  points.setZero();
  Index const O = 4;
  Basis<Cx>   basis(O, 1, M);
  basis.setZero();
  Index const P = M / O;
  for (Index ii = 0; ii < O; ii++) {
    for (Index ij = 0; ij < P; ij++) {
      basis(ii, 0, (ii * P) + ij) = std::pow(-1.f, ii) / std::sqrt(P);
    }
  }
  NDFTOp<1> ndft(points, 1, Sz1{N}, basis);
  Cx3       ks(ndft.oshape);
  ks.setConstant(1.f);
  Cx3 img(ndft.ishape);
  img.setZero();
  img = ndft.adjoint(ks);
  ks = ndft.forward(img);
  INFO("BASIS\n" << basis);
  INFO("IMG\n" << img);
  INFO("KS\n" << ks);
  CHECK(std::real(ks(0, 0, 0)) == Approx(1.f).margin(2.e-2f));
}
