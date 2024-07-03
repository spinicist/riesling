#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include <catch2/benchmark/catch_benchmark_all.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include "kernel/expsemi.hpp"
#include "kernel/radial.hpp"
#include "log.hpp"

using namespace rl;

TEST_CASE("Kernels", "[kernels]")
{
  Log::SetLevel(Log::Level::Testing);
  rl::Radial<Cx, 3, rl::ExpSemi<3>> es(2.f);
  std::array<int16_t, 3> const      c{0, 0, 0};
  auto const                        p = rl::Radial<float, 3, rl::ExpSemi<3>>::Point::Constant(0.5f);
  Sz3 const                         mc{-2, -2, -2};

  Index const B = GENERATE(1, 4, 8, 32, 128, 256);
  Cx1         b(B);
  b.setRandom();
  Cx5 x(8, B, 16, 16, 16);
  x.setRandom();
  Cx1    y(8);
  Cx1Map ym(y.data(), Sz1{8});
  BENCHMARK(fmt::format("ES Spread {}", B)) { es.spread(c, p, mc, b, y, x); };
  BENCHMARK(fmt::format("ES Gather {}", B)) { es.gather(c, p, mc, b, x, ym); };
}
