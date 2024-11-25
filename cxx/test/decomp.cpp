#include "rl/algo/decomp.hpp"
#include "rl/algo/stats.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("decomp")
{
  Index const nvar = 64;
  Index const nsamp = 256;

  SECTION("PCA")
  {
    // Due to how channels are stored, we put each sample in a column instead of a row
    Eigen::MatrixXcf data(nvar, nsamp);
    data.setRandom();
    auto cov = Covariance(data);
    Eig<Cx> eig(cov);
  }
}