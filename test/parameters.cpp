#include "log.h"
#include "sim/parameter.hpp"
#include <catch2/catch.hpp>

TEST_CASE("parameters", "")
{
  rl::Parameter A{10, 5.0, 5.0, 15.0};
  rl::Parameter B{20, 5.0, 15.0, 25.0};
  rl::Tissue T({A, B});

  SECTION("Tissue")
  {
    Eigen::ArrayXXf const p = T.values(32);
    CHECK(p.rows() == 2);
    CHECK(p.cols() == 32);
    CHECK((p.row(0) >= 5.0).all());
    CHECK((p.row(0) <= 15.0).all());
    CHECK((p.row(1) >= 15.0).all());
    CHECK((p.row(1) <= 25.0).all());
  }
}
