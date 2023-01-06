#include "log.hpp"
#include "sim/parameter.hpp"
#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

using namespace Catch;

TEST_CASE("Parameters", "[pars]")
{
  auto const t1 = rl::Parameters::T1(2048, {0.5f}, {4.3f});
  auto const t1t2b1 = rl::Parameters::T1T2B1(2048, {0.5f, 0.04f, 0.7f}, {4.3f, 2.f, 1.3f});

  SECTION("Basic")
  {
    CHECK(t1.rows() == 1);
    CHECK(t1t2b1.rows() == 3);
    CHECK(t1.cols() == 2048);
    CHECK(t1t2b1.cols() <= 2048);
  }
}
