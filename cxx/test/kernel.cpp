#include "rl/kernel/expsemi.hpp"
#include "rl/kernel/kernel.hpp"
#include "rl/kernel/tophat.hpp"
#include "rl/log.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace Catch;

TEMPLATE_TEST_CASE("TopHat-1D", "[kernels]", (rl::Kernel<float, 1, rl::TopHat<1>>))
{
  TestType                 kernel(2.f);
  typename TestType::Point p;
  p.setConstant(0.f);
  auto const k0 = kernel(p);
  INFO("W" << TestType::Width << " PW" << TestType::FullWidth << " k0 " << rl::Transpose(k0));
  CHECK(rl::Norm<false>(k0) == Approx(1.f).margin(1.e-9));
  CHECK(k0(0) == Approx(1.f).margin(1.e-6));
  p.setConstant(0.5f);
  auto const k1 = kernel(p);
  INFO("k1 " << rl::Transpose(k1));
  CHECK(k1(0) == Approx(1.f).margin(1.e-6));
  if (TestType::FullWidth > 1) { CHECK(k1(1) == Approx(k1(TestType::FullWidth - 1)).margin(1.e-5)); }
}

TEMPLATE_TEST_CASE("TopHat-2D", "[kernels]", (rl::Kernel<float, 2, rl::TopHat<1>>))
{
  TestType                 kernel(2.f);
  typename TestType::Point p;
  p.setConstant(0.f);
  auto const k0 = kernel(p);
  INFO(k0);
  CHECK(rl::Norm<false>(k0) == Approx(1.f).margin(1.e-9));
  CHECK(k0(0, 0) == Approx(1.f).margin(1.e-6));
  p.setConstant(0.5f);
  auto const k1 = kernel(p);
  INFO(k1);
  CHECK(k1(0, 0) == Approx(1.f).margin(1.e-6));
}

TEMPLATE_TEST_CASE("TopHat-3D", "[kernels]", (rl::Kernel<float, 3, rl::TopHat<1>>))
{
  TestType                 kernel(2.f);
  typename TestType::Point p;
  p.setConstant(0.f);
  auto const k0 = kernel(p);
  INFO(k0);
  CHECK(rl::Norm<false>(k0) == Approx(1.f).margin(1.e-9));
  CHECK(k0(0, 0, 0) == Approx(1.f).margin(1.e-6));
  p.setConstant(0.5f);
  auto const k1 = kernel(p);
  INFO(k1);
  CHECK(k1(0, 0, 0) == Approx(1.f).margin(1.e-6));
}

TEMPLATE_TEST_CASE("ExpSemi-1D", "[kernels]", (rl::Kernel<float, 1, rl::ExpSemi<2>>), (rl::Kernel<float, 1, rl::ExpSemi<4>>))
{
  TestType                 kernel(2.f);
  typename TestType::Point p;
  p.setConstant(0.f);
  auto const k0 = kernel(p);
  INFO("W" << TestType::Width << " PW" << TestType::FullWidth << " k0 " << rl::Transpose(k0));
  CHECK(rl::Norm<false>(k0) == Approx(1.f).margin(1.e-9));
  CHECK(k0(0) < 1.e-2f);
  CHECK(k0(0) == Approx(k0(TestType::FullWidth - 1)).margin(1.e-5));
  p.setConstant(0.5f);
  auto const k1 = kernel(p);
  INFO("k1 " << rl::Transpose(k1));
  CHECK(k1(0) == Approx(0.f).margin(1.e-9));
  CHECK(k1(1) == Approx(k1(TestType::FullWidth - 1)).margin(1.e-5));
}

TEMPLATE_TEST_CASE("ExpSemi-2D", "[kernels]", (rl::Kernel<float, 2, rl::ExpSemi<2>>), (rl::Kernel<float, 2, rl::ExpSemi<4>>))
{
  TestType                 kernel(2.f);
  typename TestType::Point p;
  p.setConstant(0.f);
  auto const k0 = kernel(p);
  INFO(k0);
  CHECK(rl::Norm<false>(k0) == Approx(1.f).margin(1.e-9));
  CHECK(k0(0, 0) < 1.e-2f);
  CHECK(k0(0, 0) == Approx(k0(TestType::FullWidth - 1, TestType::FullWidth - 1)).margin(1.e-5));
  p.setConstant(0.5f);
  auto const k1 = kernel(p);
  INFO(k1);
  CHECK(k1(0, 0) == Approx(0.f).margin(1.e-9));
  CHECK(k1(1, 1) == Approx(k1(TestType::FullWidth - 1, TestType::FullWidth - 1)).margin(1.e-5));
}

TEMPLATE_TEST_CASE("ExpSemi-3D", "[kernels]", (rl::Kernel<float, 3, rl::ExpSemi<2>>), (rl::Kernel<float, 3, rl::ExpSemi<4>>))
{
  TestType                 kernel(2.f);
  typename TestType::Point p;
  p.setConstant(0.f);
  auto const k0 = kernel(p);
  INFO(k0);
  CHECK(rl::Norm<false>(k0) == Approx(1.f).margin(1.e-9));
  CHECK(k0(0, 0, 0) < 1.e-2f);
  CHECK(k0(0, 0, 0) == Approx(k0(TestType::FullWidth - 1, TestType::FullWidth - 1, TestType::FullWidth - 1)).margin(1.e-5));
  p.setConstant(0.5f);
  auto const k1 = kernel(p);
  INFO(k1);
  CHECK(k1(0, 0, 0) == Approx(0.f).margin(1.e-9));
  CHECK(k1(1, 1, 1) == Approx(k1(TestType::FullWidth - 1, TestType::FullWidth - 1, TestType::FullWidth - 1)).margin(1.e-5));
}
