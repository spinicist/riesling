#include "log.hpp"
#include "sim/parameter.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace Catch;

TEST_CASE("Parameters", "[pars]")
{
  auto const t1 = rl::Parameters::T1(2048);
  auto const t1t2 = rl::Parameters::T1T2(2048);

  SECTION("Basic")
  {
    CHECK(t1.rows() == 1);
    CHECK(t1t2.rows() == 2);
    CHECK(t1.cols() == 2048);
    CHECK(t1t2.cols() <= 2048);
  }
}
