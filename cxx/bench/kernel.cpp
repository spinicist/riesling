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
  Radial<Cx, 3, ExpSemi<3>> es3(2.f);
  Radial<Cx, 3, ExpSemi<5>> es5(2.f);
  auto const                p = Radial<Cx, 3, ExpSemi<3>>::Point::Constant(0.5f);

  Index const B = GENERATE(1, 256); //, 4, 8, 32, 128, 256);
  Index const C = 8;
  Cx1         b(B);
  b.setRandom();
  Cx5 x(B, C, 16, 16, 16);
  x.setRandom();
  Cx1                          y(C);
  Cx1Map                       ym(y.data(), Sz1{8});
  std::array<int16_t, 3> const c{8, 8, 8};
  // auto const                   d3 = es3.dist2(p);
  // BENCHMARK(fmt::format("ES3 Fixed Spread {}", B)) { FixedKernel<Cx, 3, ExpSemi<3>>::Spread(c, d3, b, y, x); };
  float const β = ExpSemi<3>::β(2.f);
  BENCHMARK(fmt::format("ES3 Spread {}", B)) { es3.spread(c, p, b, y, x); };
  BENCHMARK(fmt::format("ES3 Fixed Spread2 {}", B)) { FixedKernel<Cx, 3, ExpSemi<3>>::Spread2(β, c, p, b, y, x); };
  // BENCHMARK(fmt::format("ES3 Fixed Gather {}", B)) { FixedKernel<Cx, 3, ExpSemi<3>>::Gather(c, d3, b, x, ym); };
  
  BENCHMARK(fmt::format("ES3 Gather {}", B)) { es3.gather(c, p, b, x, ym); };
  BENCHMARK(fmt::format("ES3 Fixed Gather2 {}", B)) { FixedKernel<Cx, 3, ExpSemi<3>>::Gather2(β, c, p, b, x, ym); };

  // auto const d5 = es5.dist2(p);
  // BENCHMARK(fmt::format("ES5 Fixed Spread {}", B)) { FixedKernel<Cx, 3, ExpSemi<5>>::Spread(c, d5, b, y, x); };
  BENCHMARK(fmt::format("ES5 Spread {}", B)) { es5.spread(c, p, b, y, x); };
  BENCHMARK(fmt::format("ES5 Fixed Spread2 {}", B)) { FixedKernel<Cx, 3, ExpSemi<5>>::Spread2(β, c, p, b, y, x); };
  // BENCHMARK(fmt::format("ES5 Fixed Gather {}", B)) { FixedKernel<Cx, 3, ExpSemi<5>>::Gather(c, d5, b, x, ym); };
  BENCHMARK(fmt::format("ES5 Gather {}", B)) { es5.gather(c, p, b, x, ym); };
  BENCHMARK(fmt::format("ES5 Fixed Gather2 {}", B)) { FixedKernel<Cx, 3, ExpSemi<5>>::Gather2(β, c, p, b, x, ym); };
}
