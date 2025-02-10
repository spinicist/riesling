#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "rl/kernel/kernel.hpp"
#include "rl/log.hpp"

using namespace rl;

TEMPLATE_TEST_CASE(
  "Kernels", "[kernels]", (Kernel<Cx, 3, ExpSemi<2>>), (Kernel<Cx, 3, ExpSemi<4>>), (Kernel<Cx, 3, ExpSemi<6>>))
{
  TestType   k(2.f);
  auto const p = TestType::Point::Constant(0.5f);

  Index const B = 8;
  Index const C = 8;
  Cx1         b(B);
  b.setRandom();
  Cx5 x(16, 16, 16, C, B);
  x.setRandom();
  Cx1 y(C);
  y.setRandom();

  Eigen::Array<int16_t, 3, 1> const c{8, 8, 8};

  BENCHMARK(fmt::format("ES{} Spread", TestType::Width, B)) { k.spread(c, p, y, x); };
  BENCHMARK(fmt::format("ES{} Gather", TestType::Width, B)) { k.gather(c, p, x, y); };

  BENCHMARK(fmt::format("ES{} Spread {}", TestType::Width, B)) { k.spread(c, p, b, y, x); };
  BENCHMARK(fmt::format("ES{} Gather {}", TestType::Width, B)) { k.gather(c, p, b, x, y); };
}
