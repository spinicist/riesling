#include "../src/gridder.h"
#include "../src/log.h"
#include <catch2/catch.hpp>

TEST_CASE("Gridder with single point", "GRID-SINGLE")
{
  Log log(false);
  RadialInfo info{
      .matrix = {2, 2, 2},
      .voxel_size = {1, 1, 1},
      .read_points = 1,
      .read_gap = 0,
      .spokes_hi = 1,
      .spokes_lo = 0,
      .lo_scale = 1,
      .channels = 1,
      .volumes = 1,
  };
  R3 traj(3, 1, 1);
  traj.setZero();

  SECTION("NN")
  {
    Gridder gridder(info, traj, 2, false, false, false, log);
    gridder.setDC(1.f);
    Cx2 rad(info.read_points, info.spokes_total());
    CHECK(rad.dimension(0) == 1);
    CHECK(rad.dimension(1) == 1);
    Cx3 cart = gridder.newGrid1();
    CHECK(cart.dimension(0) == 4);
    CHECK(cart.dimension(1) == 4);
    CHECK(cart.dimension(2) == 4);
    rad.setConstant(1.f);
    cart.setZero();
    gridder.toCartesian(rad, cart);
    CHECK(cart(0, 0, 0).real() == Approx(1.f));
    gridder.toRadial(cart, rad);
    CHECK(rad(0, 0).real() == Approx(1.f));
  }

  SECTION("KB Estimate")
  {
    Gridder gridder(info, traj, 2, true, true, false, log);
    Cx2 rad(info.read_points, info.spokes_total());
    CHECK(rad.dimension(0) == 1);
    CHECK(rad.dimension(1) == 1);
    Cx3 cart = gridder.newGrid1();
    CHECK(cart.dimension(0) == 4);
    CHECK(cart.dimension(1) == 4);
    CHECK(cart.dimension(2) == 4);
    rad.setConstant(1.f);
    cart.setZero();
    gridder.toCartesian(rad, cart);
    gridder.toRadial(cart, rad);
    CHECK(rad(0, 0).real() == Approx(1.f));
  }
}

TEST_CASE("Gridder with single spoke", "GRID-SPOKE")
{
  Log log(false);
  RadialInfo info{
      .matrix = {4, 4, 4},
      .voxel_size = {1, 1, 1},
      .read_points = 4,
      .read_gap = 0,
      .spokes_hi = 1,
      .spokes_lo = 0,
      .lo_scale = 1,
      .channels = 1,
      .volumes = 1,
  };
  R3 traj(3, info.read_points, info.spokes_total());
  traj.setZero();
  traj(0, 1, 0) = 1.f / 3.f;
  traj(0, 2, 0) = 2.f / 3.f;
  traj(0, 3, 0) = 1.f;

  Cx2 rad(info.read_points, info.spokes_total());
  SECTION("NN")
  {
    Gridder gridder(info, traj, 2, false, false, false, log);
    gridder.setDC(1.f);
    Cx3 cart = gridder.newGrid1();
    CHECK(cart.dimension(0) == 8);
    CHECK(cart.dimension(1) == 8);
    CHECK(cart.dimension(2) == 8);
    rad.setConstant(1.f);
    cart.setZero();
    gridder.toCartesian(rad, cart);
    CHECK(cart(0, 0, 0).real() == Approx(1.f));
    CHECK(cart(1, 0, 0).real() == Approx(1.f));
    CHECK(cart(2, 0, 0).real() == Approx(1.f));
    CHECK(cart(3, 0, 0).real() == Approx(1.f));
    CHECK(cart(4, 0, 0).real() == Approx(0.f));
    CHECK(cart(0, 1, 0).real() == Approx(0.f));
    CHECK(cart(0, 0, 1).real() == Approx(0.f));
    gridder.toRadial(cart, rad);
    CHECK(rad(0, 0).real() == Approx(1.f));
  }

  SECTION("KB Estimate")
  {
    Gridder gridder(info, traj, 2, true, true, false, log);
    Cx3 cart = gridder.newGrid1();
    CHECK(cart.dimension(0) == 8);
    CHECK(cart.dimension(1) == 8);
    CHECK(cart.dimension(2) == 8);
    rad.setConstant(1.f);
    cart.setZero();
    gridder.toCartesian(rad, cart);
    gridder.toRadial(cart, rad);
    CHECK(rad(0, 0).real() == Approx(1.f).margin(1.e-2f));
  }
}
