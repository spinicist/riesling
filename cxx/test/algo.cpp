#include "rl/algo/lsmr.hpp"
#include "rl/op/ops.hpp"
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
  Eigen::VectorXcf const x = Eigen::ArrayXf::LinSpaced(N, 0, N - 1).cast<Cx>();
  auto y = A->forward(x);

  SECTION("LSMR")
  {
    LSMR lsmr{A, M};
    auto xx = lsmr.run(y);
    INFO("x " << x.transpose() << "\ny " << y.transpose() << "\nxx " << xx.transpose());
    CHECK((x - xx).stableNorm() == Approx(0.f).margin(1.e-3f));
  }
}