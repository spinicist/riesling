#include "rl/op/nufft-decant.hpp"
#include "rl/kernel/tophat.hpp"
#include "rl/log/log.hpp"
#include "rl/op/grid-decant.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Decant1", "[op]")
{
  // Log::SetDisplayLevel(Log::Display::High);
  Threads::SetGlobalThreadCount(1);
  Index const M = GENERATE(16, 32);
  auto const  matrix = Sz1{M};
  Re3         points(1, M, 1);
  points.setZero();
  for (Index ii = 0; ii < M; ii++) {
    points(0, ii, 0) = -M / 2.f + ii;
  }
  TrajectoryN<1> const traj(points, matrix);

  Cx3 sk(1, 1, 2);
  sk.setZero();
  sk(0, 0, 0) = Cx(1.f / std::sqrt(2.f), 0.f);
  sk(0, 0, 1) = Cx(0.f, 1.f / std::sqrt(2.f));

  float const osamp = GENERATE(2.f);
  auto        nufft = TOps::NUFFTDecant<1, ExpSemi<6>>::Make(GridOpts<1>{.osamp = osamp}, traj, sk, nullptr);

  Cx3 noncart(nufft->oshape);
  Cx2 cart(nufft->ishape);
  noncart.setZero();
  cart.setConstant(1.f);
  INFO("M " << M << " OS " << osamp);
  noncart = nufft->forward(cart);
  INFO("noncart\n" << noncart);
  auto const cs = std::sqrt(cart.size());
  CHECK((Norm<false>(cart) - Norm<false>(noncart)) / cs == Approx(0.f).margin(1e-4f));
  CHECK(std::abs(noncart(0, M / 2, 0)) == Approx(std::sqrt(M / 2.f)).margin(1e-4f));
  CHECK(std::abs(noncart(1, M / 2, 0)) == Approx(std::sqrt(M / 2.f)).margin(1e-4f));
  CHECK(std::abs(noncart(0, 0, 0)) == Approx(0.f).margin(1e-4f));
  cart = nufft->adjoint(noncart);
  INFO("cart\n" << cart);
  CHECK((Norm<false>(cart) - Norm<false>(noncart)) / cs == Approx(0.f).margin(1e-4f));
  for (Index ij = 0; ij < cart.dimension(1); ij++) {
    for (Index ii = 0; ii < cart.dimension(0); ii++) {
      CHECK(std::real(cart(ii, ij)) == Approx(1.f).margin(1e-4f));
    }
  }
}
