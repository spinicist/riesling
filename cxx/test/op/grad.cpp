#include "rl/op/grad.hpp"
#include "rl/tensors.hpp"
#include "rl/algo/eig.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("grad-eig", "[op]")
{
  auto sz = GENERATE(32, 48, 64);
  SECTION("Eigenvalues")
  {
    Cx1 x(sz);
    Index const N = 64;
    auto g1 = TOps::Grad<1, 1>::Make(Sz1{sz}, Sz1{0}, 0);
    auto const [v1, vc1] = PowerMethodForward(g1, nullptr, N);
    auto g2 = TOps::Grad<2, 2>::Make(Sz2{sz, sz}, Sz2{0, 1}, 0);
    auto const [v2, vc2] = PowerMethodForward(g2, nullptr, N);
    auto g3 = TOps::Grad<3, 3>::Make(Sz3{sz, sz, sz}, Sz3{0, 1, 2}, 0);
    auto const [v3, vc3] = PowerMethodForward(g3, nullptr, N);

    CHECK(v1 == Approx(1.f).margin(3.e-2f));
    CHECK(v2 == Approx(1.f).margin(3.e-2f));
    CHECK(v3 == Approx(1.f).margin(3.e-2f));
  }
}