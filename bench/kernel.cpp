#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include "kernel/expsemi.hpp"
#include "kernel/kaiser.hpp"
#include "kernel/nn.hpp"
#include "kernel/radial.hpp"
#include "kernel/rectilinear.hpp"
#include "kernel/triangle.hpp"
#include "types.h"

using namespace rl;

TEMPLATE_TEST_CASE(
  "Nearest Neighbour", "[NN]", (rl::NearestNeighbour<1>), (rl::NearestNeighbour<2>), (rl::NearestNeighbour<3>))
{
  TestType k(2.f);
  typename TestType::Point p;
  p.setConstant(0.5f);
  BENCHMARK("NN")
  {
    k(p);
  };
}

TEST_CASE(
  "Kernels", "[kernels]")
{
  rl::Radial<3, rl::KaiserBessel<7>> kb(2.f);
  rl::Radial<3, rl::ExpSemi<7>> es(2.f);
  typename rl::Radial<3, rl::KaiserBessel<7>>::Point p;
  p.setConstant(0.5f);
  BENCHMARK("KB")
  {
    kb(p);
  };
  BENCHMARK("ES")
  {
    es(p);
  };
}
