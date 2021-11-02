#include "../src/log.h"
#include "../src/sim.h"
#include <catch2/catch.hpp>

TEST_CASE("SIM-Parameters", "[SIM]")
{
  Sim::Parameter A{10, 0.1, 1.0, false};
  Sim::Parameter B{10, 1.0, 10.0, false};
  Sim::Parameter C{10, 10.0, 100.0, false};

  SECTION("Generator")
  {
    Sim::ParameterGenerator<3> gen{{A, B, C}};

    CHECK(gen.totalN() == (A.N * B.N * C.N));
    Eigen::Array3f p{0.1, 1.0, 10.};
    CHECK(!(gen.values(0) - p).any());
    p(1) = 2.0;
    CHECK(!(gen.values(10) - p).any());
    p(0) = 0.2;
    CHECK(!(gen.values(11) - p).any());
    p(2) = 20.0;
    CHECK(!(gen.values(111) - p).any());
  }
}
