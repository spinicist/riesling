#include "log.hpp"
#include "sim/parameter.hpp"
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace Catch;

TEST_CASE("Parameters", "[pars]")
{
  auto const t1t2pd =
    rl::ParameterGrid(3, Eigen::Array3f{0.5f, 0.04f, 0.7f}, Eigen::Array3f{4.0f, 0.1f, 1.f}, Eigen::Array3f{0.5f, 0.02f, 0.1f});

  SECTION("Basic")
  {
    Index const nT = 8 * 4 * 4;
    INFO(t1t2pd.transpose());
    CHECK(t1t2pd.rows() == 3);
    CHECK(t1t2pd.cols() == nT);
    CHECK(t1t2pd(0, 0) == Approx(0.5f));
    CHECK(t1t2pd(1, 0) == Approx(0.04f));
    CHECK(t1t2pd(2, 0) == Approx(0.7f));
    CHECK(t1t2pd(0, nT - 1) == Approx(4.0f));
    CHECK(t1t2pd(1, nT - 1) == Approx(0.1f));
    CHECK(t1t2pd(2, nT - 1) == Approx(1.0f));
  }
}
