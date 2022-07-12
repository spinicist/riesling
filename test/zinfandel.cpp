#include "../src/log.h"
#include "../src/tensorOps.h"
#include "../src/zin-grappa.hpp"
#include <catch2/catch.hpp>
#include <fmt/format.h>

using namespace rl;

TEST_CASE("zinfandel-data")
{
  Index const n_readpoints = 16;
  Index const n_spoke = 1;
  Index const n_coil = 4;
  float const scale = 2.f;
  Cx3 kspace(n_coil, n_readpoints, n_spoke);
  kspace.setZero();

  for (auto is = 0; is < n_spoke; is++) {
    for (auto ir = 0; ir < n_readpoints; ir++) {
      for (auto ic = 0; ic < n_coil; ic++) {
        kspace(ic, ir, is) = std::complex<float>(is + ir + ic + 1, 0);
      }
    }
  }

  SECTION("Grab Sources")
  {
    Index const n_src = 4;
    Index const n_read = 4;
    auto const S = GrabSources(kspace, scale, n_src, 0, n_read, {0});
    CHECK(S.rows() == (n_coil * n_src));
    CHECK(S.cols() == n_read);
    CHECK((S.array().real() > 0.).all());
    CHECK(S(0, 0).real() == Approx(1. / scale));
    CHECK(
      S((n_coil * n_src) - 1, n_read - 1).real() ==
      Approx((n_read - 1 + n_src - 1 + n_coil - 1 + 1) / scale));
  }

  SECTION("Grab Targets")
  {
    Index const n_read = 4;
    auto const T = GrabTargets(kspace, scale, 0, n_read, {0});
    CHECK(T.rows() == n_coil);
    CHECK(T.cols() == n_read);
    CHECK((T.array().real() > 0.).all());
    CHECK(T(0, 0).real() == Approx(1. / scale));
    CHECK(T(n_coil - 1, n_read - 1).real() == Approx((n_read + n_coil - 1) / scale));
  }
}

TEST_CASE("zinfandel-algo")
{
  Index const n_read = 12;
  Index const n_spoke = 4;
  Index const n_coil = 4;
  Cx3 kspace(n_coil, n_read, n_spoke);
  kspace.setZero();
  for (auto is = 0; is < n_spoke; is++) {
    for (auto ir = 0; ir < n_read; ir++) {
      for (auto ic = 0; ic < n_coil; ic++) {
        kspace(ic, ir, is) = std::complex<float>(ir + ic + is + 1, 0);
      }
    }
  }
  R3 traj(3, n_read, n_spoke);
  traj.setZero();
  for (auto is = 0; is < n_spoke; is++) {
    R1 endPoint(3);
    endPoint(0) = 0.f;
    endPoint(1) = cos(is * M_PI / n_spoke);
    endPoint(2) = sin(is * M_PI / n_spoke);
    for (auto ir = 0; ir < n_read; ir++) {
      traj.chip(is, 2).chip(ir, 1) = endPoint * (1.f * ir / n_read);
    }
  }

  SECTION("Run")
  {
    Index const n_gap = 2;
    Cx3 test_kspace = kspace;
    test_kspace.slice(Sz3{0, 0, 0}, Sz3{n_coil, n_gap, n_spoke}).setZero();
    zinGRAPPA(n_gap, 2, 1, 4, 0.0, traj, test_kspace);
    Cx3 diff = test_kspace.slice(Sz3{0, 0, 0}, Sz3{n_coil, n_gap, n_spoke}) -
               kspace.slice(Sz3{0, 0, 0}, Sz3{n_coil, n_gap, n_spoke});
    float const sum_diff = Norm(diff) / diff.size();
    CHECK(sum_diff == Approx(0.f).margin(1.e-4f));
  }
}