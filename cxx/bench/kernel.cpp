#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "rl/kernel/kernel.hpp"
#include "rl/log/log.hpp"

using namespace rl;

TEMPLATE_TEST_CASE("Kernels", "[kernels]", (Kernel<3, ExpSemi<2>>), (Kernel<3, ExpSemi<4>>), (Kernel<3, ExpSemi<6>>))
{
  TestType   k(2.f);
  auto const p = TestType::Point::Constant(0.5f);

  BENCHMARK(fmt::format("ES{} Spread", TestType::Width)) { k(p); };
  BENCHMARK(fmt::format("ES{} Gather", TestType::Width)) { k(p); };
}
