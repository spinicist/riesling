#include "rl/algo/lsmr.hpp"
#include "rl/log/log.hpp"
#include "rl/algo/admm.hpp"
#include "rl/op/ops.hpp"
#include "rl/prox/norms.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Algorithms", "[alg]")
{
  Index const N = 8;
  Eigen::MatrixXf Amat = Eigen::MatrixXf::Identity(N, N) + Eigen::MatrixXf::Ones(N, N);
  auto const A = std::make_shared<Ops::MatMul>(Amat);
  auto const M = Ops::Identity::Make(N);
  /* Sparse so we can use L1 reg below */
  Eigen::VectorXcf x = Eigen::ArrayXf::Zero(N).cast<Cx>();
  x(N / 2) = 1.f;
  auto y = A->forward(x);

  // rl::Log::SetDisplayLevel(rl::Log::Display::Low);
  SECTION("LSMR")
  {
    LSMR lsmr{A, M};
    auto xx = lsmr.run(y);
    INFO("x " << x.transpose() << "\ny " << y.transpose() << "\nxx " << xx.transpose());
    CHECK((x - xx).stableNorm() == Approx(0.f).margin(1.e-3f));
  }

  y += Eigen::ArrayXf::Random(y.rows()).cast<Cx>().matrix() * 1e-3f;
  SECTION("ADMM")
  {
    std::vector<Regularizer> reg;
    reg.push_back(Regularizer{nullptr, Proxs::L2<1, 1>::Make(1.e-3f * y.norm(), Sz1{A->rows()}, Sz1{0})});
    ADMM admm{A, M, reg, ADMM::Opts{}};
    auto xx = admm.run(y);
    INFO("x " << x.transpose() << "\ny " << y.transpose() << "\nxx " << xx.transpose());
    CHECK((x - xx).stableNorm() == Approx(0.f).margin(1.e-2f));
  }
}
