#include "rl/patches.hpp"
#include "rl/log/log.hpp"
#include "rl/prox/llr.hpp"
#include "rl/tensors.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("Patches", "[patch]")
{
  Index const M = 16;
  Index const P = 3, W = 3;
  Sz5 const   shape{M, M, M, 2, 1};
  Cx5         x(shape), z(shape);

  x.chip<3>(0).setConstant(1.f);
  x.chip<3>(1).setConstant(2.f);

  SECTION("Patches")
  {
    auto flip = [](Cx5 const &x) -> Cx5 { return -x; };
    Patches(P, W, false, flip, x, z);
    bool all_equal = B0((z == -x).all())();
    CHECK(all_equal);
  }
}
