#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <Eigen/Dense>
#include <catch2/catch.hpp>

#include "types.h"
#include "kernel/rectilinear.hpp"
#include "kernel/radial.hpp"
#include "kernel/expsemi.hpp"
#include "kernel/kaiser.hpp"
#include "kernel/triangle.hpp"
#include "kernel/nn.hpp"

using namespace rl;

TEMPLATE_TEST_CASE(
  "Nearest Neighbour",
  "[kernels]",
  (rl::NearestNeighbour<1>),
  (rl::NearestNeighbour<2>),
  (rl::NearestNeighbour<3>))
{
  TestType k(2.f);
  typename TestType::Point p;
  p.setConstant(0.5f);
  BENCHMARK("NN") {
    k(p);
  };
}

TEMPLATE_TEST_CASE(
  "1D Kernels",
  "[kernels]",
  (rl::Triangle<3>),
  (rl::KaiserBessel<3>),
  (rl::ExpSemi<3>))
{
  rl::Rectilinear<1, TestType> rect(2.f);
  rl::Radial<1, TestType> rad(2.f);
  typename rl::Radial<1, TestType>::Point p;
  p.setConstant(0.5f);
  BENCHMARK("Rectilinear") {
    rect(p);
  };
  BENCHMARK("Radial") {
    rad(p);
  };
}

TEMPLATE_TEST_CASE(
  "2D Kernels",
  "[kernels]",
  (rl::Triangle<3>),
  (rl::KaiserBessel<3>),
  (rl::ExpSemi<3>))
{
  rl::Rectilinear<2, TestType> rect(2.f);
  rl::Radial<2, TestType> rad(2.f);
  typename rl::Radial<2, TestType>::Point p;
  p.setConstant(0.5f);
  BENCHMARK("Rectilinear") {
    rect(p);
  };
  BENCHMARK("Radial") {
    rad(p);
  };
}

TEMPLATE_TEST_CASE(
  "3D Kernels",
  "[kernels]",
  (rl::Triangle<3>),
  (rl::KaiserBessel<3>),
  (rl::ExpSemi<3>))
{
  rl::Rectilinear<3, TestType> rect(2.f);
  rl::Radial<3, TestType> rad(2.f);
  typename rl::Radial<3, TestType>::Point p;
  p.setConstant(0.5f);
  BENCHMARK("Rectilinear") {
    rect(p);
  };
  BENCHMARK("Radial") {
    rad(p);
  };
}
