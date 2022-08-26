#include "../src/kernel-fi.hpp"
#include "../src/kernel-kb.hpp"
#include "../src/kernel-nn.hpp"
#include "../src/newkernel.hpp"
#include "../src/types.h"
#include <catch2/catch.hpp>

#include "log.h"

TEST_CASE("New Kernels", "[kernel]")
{
  // rl::NewSizedKernel<2, 3> test;
  // auto k1 = rl::NewFlatIron<1, 3>(2.f);
  // fmt::print("k1\n {}\n", k1.k(Eigen::Matrix<float, 1, 1>(0)));

  auto k2 = rl::NewFlatIron<2, 7>(2.f);
  fmt::print("k2\n {}\n", k2.k(Eigen::Vector2f(0, 0)));
  fmt::print("k2\n {}\n", k2.k(Eigen::Vector2f(0.5, 0)));
  // auto k3 = rl::NewFlatIron<3, 3>(2.f);
  // fmt::print("k2\n {}\n", k3.k(Eigen::Vector3f(0)));
}

// TEST_CASE("Kernels - NearestNeighbour", "[Kernel][NN]")
// {
//   auto const nn = rl::NearestNeighbour();
//   auto const k = nn.k(rl::Point3{0.f, 0.f, 0.f});
//   CHECK(k(0, 0, 0) == Approx(1.f).margin(1.e-5));
//   CHECK(Norm(k) == Approx(1.f).margin(1.e-9));
// }

// TEMPLATE_TEST_CASE(
//   "Kernels 2D", "[Kernel][KB]", (rl::KaiserBessel<3, 1>), (rl::KaiserBessel<5, 1>), (rl::KaiserBessel<7, 1>))
// {
//   TestType kernel(2.f);

//   SECTION("Center")
//   {
//     auto const k = kernel.k(rl::Point3{0.f, 0.f, 0.f});
//     CHECK(Norm(k) == Approx(1.f).margin(1.e-9));
//     CHECK(k(0, kernel.inPlane() / 2, 0) == k(kernel.inPlane() / 2, 0, 0));
//   }

//   SECTION("Off-center")
//   {
//     auto const k2 = kernel.k(rl::Point3{-0.5f, 0.f, 0.f});
//     CHECK(Sum(k2) == Approx(1.f).margin(1.e-2));
//     CHECK(k2(kernel.inPlane() / 2 - 1, kernel.inPlane() / 2, 0) == k2(kernel.inPlane() / 2, kernel.inPlane() / 2,
//     0));
//   }
// }

// TEMPLATE_TEST_CASE(
//   "Kernels 3D", "[Kernel][KB]", (rl::KaiserBessel<3, 3>), (rl::KaiserBessel<5, 5>), (rl::KaiserBessel<7, 7>))
// {
//   TestType kernel(2.f);

//   SECTION("Center")
//   {
//     auto const k = kernel.k(rl::Point3{0.f, 0.f, 0.f});
//     CHECK(Norm(k) == Approx(1.f).margin(1.e-9));
//     CHECK(k(0, kernel.inPlane() / 2, 0) == k(kernel.inPlane() / 2, 0, 0));
//   }

//   SECTION("Off-center")
//   {
//     auto const k2 = kernel.k(rl::Point3{0.f, 0.f, -0.5f});
//     CHECK(Norm(k2) == Approx(1.f).margin(5.e-2));
//     CHECK(k2(0, kernel.inPlane() / 2, kernel.inPlane() / 2 - 1) == k2(0, kernel.inPlane() / 2, kernel.inPlane() /
//     2));
//   }
// }

// TEMPLATE_TEST_CASE(
//   "Kernels 2D", "[Kernel][FI]", (rl::FlatIron<3, 1>), (rl::FlatIron<5, 1>), (rl::FlatIron<7, 1>))
// {
//   TestType kernel(2.f);

//   SECTION("Center")
//   {
//     auto const k = kernel.k(rl::Point3{0.f, 0.f, 0.f});
//     CHECK(Norm(k) == Approx(1.f).margin(1.e-9));
//     CHECK(k(0, kernel.inPlane() / 2, 0) == k(kernel.inPlane() / 2, 0, 0));
//   }

//   SECTION("Off-center")
//   {
//     auto const k2 = kernel.k(rl::Point3{-0.5f, 0.f, 0.f});
//     CHECK(Norm(k2) == Approx(1.f).margin(5.e-2));
//     CHECK(k2(kernel.inPlane() / 2 - 1, kernel.inPlane() / 2, 0) == k2(kernel.inPlane() / 2, kernel.inPlane() / 2,
//     0));
//   }
// }

// TEMPLATE_TEST_CASE(
//   "Kernels 3D", "[Kernel][FI]", (rl::FlatIron<3, 3>), (rl::FlatIron<5, 5>), (rl::FlatIron<7, 7>))
// {
//   TestType kernel(2.f);

//   SECTION("Center")
//   {
//     auto const k = kernel.k(rl::Point3{0.f, 0.f, 0.f});
//     CHECK(Norm(k) == Approx(1.f).margin(1.e-9));
//     CHECK(k(0, kernel.inPlane() / 2, 0) == k(kernel.inPlane() / 2, 0, 0));
//   }

//   SECTION("Off-center")
//   {
//     auto const k2 = kernel.k(rl::Point3{0.f, 0.f, -0.5f});
//     CHECK(Norm(k2) == Approx(1.f).margin(5.e-2));
//     CHECK(k2(0, kernel.inPlane() / 2, kernel.inPlane() / 2 - 1) == k2(0, kernel.inPlane() / 2, kernel.inPlane() /
//     2));
//   }
// }
