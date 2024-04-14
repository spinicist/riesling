#include "algo/cg.hpp"
#include "op/ops.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Algorithms", "[alg]")
{
  Index const N = 8;
  Eigen::MatrixXf Amat = Eigen::MatrixXf::Identity(N, N) + Eigen::MatrixXf::Ones(N, N);
  Amat.array() += 1.f;
  auto A = std::make_shared<Ops::MatMul<float>>(Amat);
  Eigen::VectorXf const x = Eigen::ArrayXf::LinSpaced(N, 0, N - 1);
  auto y = A->forward(x);

  SECTION("CG")
  {
    ConjugateGradients<float> cg{A};
    auto xx = cg.run(y.data());
    CHECK((x - xx).norm() == Approx(0.f).margin(1.e-3f));
  }
}