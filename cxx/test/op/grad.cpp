#include "rl/op/grad.hpp"
#include "rl/tensors.hpp"
#include "rl/algo/eig.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

TEST_CASE("grad-eig", "[eig]")
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

    auto gb1 = TOps::Grad<1, 1>::Make(Sz1{sz}, Sz1{0}, 1);
    auto const [vb1, vbc1] = PowerMethodForward(g1, nullptr, N);
    auto gb2 = TOps::Grad<2, 2>::Make(Sz2{sz, sz}, Sz2{0, 1}, 1);
    auto const [vb2, vbc2] = PowerMethodForward(g2, nullptr, N);
    auto gb3 = TOps::Grad<3, 3>::Make(Sz3{sz, sz, sz}, Sz3{0, 1, 2}, 1);
    auto const [vb3, vbc3] = PowerMethodForward(g3, nullptr, N);

    CHECK(vb1 == Approx(1.f).margin(3.e-2f));
    CHECK(vb2 == Approx(1.f).margin(3.e-2f));
    CHECK(vb3 == Approx(1.f).margin(3.e-2f));

    auto gd1 = TOps::Div<1, 1>::Make(Sz1{sz}, Sz1{0}, 0);
    auto const [vd1, vdc1] = PowerMethodForward(g1, nullptr, N);
    auto gd2 = TOps::Div<2, 2>::Make(Sz2{sz, sz}, Sz2{0, 1}, 0);
    auto const [vd2, vdc2] = PowerMethodForward(g2, nullptr, N);
    auto gd3 = TOps::Div<3, 3>::Make(Sz3{sz, sz, sz}, Sz3{0, 1, 2}, 0);
    auto const [vd3, vdc3] = PowerMethodForward(g3, nullptr, N);

    CHECK(vd1 == Approx(1.f).margin(3.e-2f));
    CHECK(vd2 == Approx(1.f).margin(3.e-2f));
    CHECK(vd3 == Approx(1.f).margin(3.e-2f));

    auto gg1 = TOps::GradVec<2, 1>::Make(Sz2{sz, 1}, Sz1{0}, 0);
    auto const [vg1, vgc1] = PowerMethodForward(g1, nullptr, N);
    auto gg2 = TOps::GradVec<3, 2>::Make(Sz3{sz, sz, 2}, Sz2{0, 1}, 0);
    auto const [vg2, vgc2] = PowerMethodForward(g2, nullptr, N);
    auto gg3 = TOps::GradVec<4, 3>::Make(Sz4{sz, sz, sz, 3}, Sz3{0, 1, 2}, 0);
    auto const [vg3, vgc3] = PowerMethodForward(g3, nullptr, N);

    CHECK(vg1 == Approx(1.f).margin(3.e-2f));
    CHECK(vg2 == Approx(1.f).margin(3.e-2f));
    CHECK(vg3 == Approx(1.f).margin(3.e-2f));
  }
}