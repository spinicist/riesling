#include "../src/log.h"
#include "../src/op/grid-kb.h"
#include "../src/op/grid-nn.h"
#include "../src/trajectory.h"
#include <catch2/catch.hpp>

TEST_CASE("Gridding", "GRIDDING")
{
  Log log;
  Info info{
      .type = Info::Type::ThreeD,
      .channels = 4,
      .matrix = {2, 2, 2},
      .read_points = 1,
      .read_gap = 0,
      .spokes_hi = 1,
      .spokes_lo = 0,
      .lo_scale = 1};
  float const osamp = 2.f;
  R3 points(3, 1, 1);
  points.setZero();
  Trajectory traj(info, points, log);

  SECTION("NN")
  {
    GridNN gridder(traj, osamp, false, log);
    Cx3 rad = info.noncartesianVolume();
    CHECK(rad.dimension(0) == 4);
    CHECK(rad.dimension(1) == 1);
    Cx4 cart = gridder.newMultichannel(info.channels);
    CHECK(cart.dimension(1) == 4);
    CHECK(cart.dimension(2) == 4);
    CHECK(cart.dimension(3) == 4);
    rad.setConstant(1.f);
    cart.setZero();
    gridder.Adj(rad, cart);
    CHECK(cart(0, 2, 2, 2).real() == Approx(1.f));
    gridder.A(cart, rad);
    CHECK(rad(0, 0, 0).real() == Approx(1.f));
  }

  SECTION("NN Multicoil")
  {
    GridNN gridder(traj, osamp, false, log);
    Cx3 rad = info.noncartesianVolume();
    CHECK(rad.dimension(0) == info.channels);
    CHECK(rad.dimension(1) == info.read_points);
    Cx4 cart = gridder.newMultichannel(info.channels);
    CHECK(cart.dimension(0) == info.channels);
    CHECK(cart.dimension(1) == 4);
    CHECK(cart.dimension(2) == 4);
    CHECK(cart.dimension(3) == 4);
    rad.setConstant(1.f);
    cart.setZero();
    gridder.Adj(rad, cart);
    CHECK(cart(0, 2, 2, 2).real() == Approx(1.f));
    CHECK(cart(1, 2, 2, 2).real() == Approx(1.f));
    CHECK(cart(2, 2, 2, 2).real() == Approx(1.f));
    CHECK(cart(3, 2, 2, 2).real() == Approx(1.f));
    gridder.A(cart, rad);
    CHECK(rad(0, 0, 0).real() == Approx(1.f));
    CHECK(rad(1, 0, 0).real() == Approx(1.f));
    CHECK(rad(2, 0, 0).real() == Approx(1.f));
    CHECK(rad(3, 0, 0).real() == Approx(1.f));
  }

  SECTION("KB Multicoil")
  {
    GridKB3D gridder(traj, osamp, false, log);
    Cx3 rad = info.noncartesianVolume();
    CHECK(rad.dimension(0) == info.channels);
    CHECK(rad.dimension(1) == info.read_points);
    Cx4 cart = gridder.newMultichannel(info.channels);
    CHECK(cart.dimension(0) == info.channels);
    CHECK(cart.dimension(1) == 4);
    CHECK(cart.dimension(2) == 4);
    CHECK(cart.dimension(3) == 4);
    rad.setConstant(1.f);
    cart.setZero();
    gridder.Adj(rad, cart);
    gridder.A(cart, rad);
    CHECK(rad(0, 0, 0).real() == Approx(0.14457f).margin(1.e-5f));
    CHECK(rad(1, 0, 0).real() == Approx(0.14457f).margin(1.e-5f));
    CHECK(rad(2, 0, 0).real() == Approx(0.14457f).margin(1.e-5f));
    CHECK(rad(3, 0, 0).real() == Approx(0.14457f).margin(1.e-5f));
  }
}

TEST_CASE("SingleSpoke", "GRIDDING")
{
  Log log;
  Info info{
      .type = Info::Type::ThreeD,
      .channels = 1,
      .matrix = {4, 4, 4},
      .read_points = 4,
      .read_gap = 0,
      .spokes_hi = 1,
      .spokes_lo = 0,
      .lo_scale = 1};
  float const osamp = 2.f;
  R3 points(3, info.read_points, info.spokes_total());
  points.setZero();
  // Trajectory points are scaled between -0.5 and 0.5
  points(0, 1, 0) = 0.5f * 1.f / 3.f;
  points(0, 2, 0) = 0.5f * 2.f / 3.f;
  points(0, 3, 0) = 0.5f * 1.f;
  Trajectory traj(info, points, log);

  SECTION("NN")
  {
    GridNN gridder(traj, osamp, false, log);
    Cx4 cart = gridder.newMultichannel(1);
    CHECK(cart.dimension(0) == 1);
    CHECK(cart.dimension(1) == 8);
    CHECK(cart.dimension(2) == 8);
    CHECK(cart.dimension(3) == 8);
    Cx3 rad(1, info.read_points, info.spokes_total());
    rad.setConstant(1.f);
    cart.setZero();
    gridder.Adj(rad, cart);
    CHECK(cart(0, 4, 4, 4).real() == Approx(1.f));
    CHECK(cart(0, 5, 4, 4).real() == Approx(1.f));
    CHECK(cart(0, 6, 4, 4).real() == Approx(1.f));
    CHECK(cart(0, 7, 4, 4).real() == Approx(1.f));
    CHECK(cart(0, 4, 5, 5).real() == Approx(0.f));
    CHECK(cart(0, 4, 5, 4).real() == Approx(0.f));
    CHECK(cart(0, 4, 4, 5).real() == Approx(0.f));
    gridder.A(cart, rad);
    CHECK(rad(0, 0, 0).real() == Approx(1.f));
  }
}
