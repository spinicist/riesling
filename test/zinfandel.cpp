#include "../src/log.h"
#include "../src/zinfandel.h"
#include <catch2/catch.hpp>
#include <fmt/format.h>

TEST_CASE("ZINFANDEL Data mangling", "[ZINFANDEL]")
{
  long const n_readpoints = 16;
  long const n_spoke = 1;
  long const n_coil = 4;
  float const scale = 2.f;
  Cx3 kspace(n_coil, n_readpoints, n_spoke);
  kspace.setZero();

  for (auto is = 0; is < n_spoke; is++) {
    for (auto ir = 0; ir < n_readpoints; ir++) {
      for (auto ic = 0; ic < n_coil; ic++) {
        kspace(ic, ir, is) = std::complex<float>(ir + ic + 1, 0);
      }
    }
  }

  SECTION("Grab Sources")
  {
    long const n_src = 4;
    long const n_read = n_readpoints - (n_src - 1);
    auto const S = GrabSources(kspace, scale, n_src, 0, 1, 0, n_read);
    CHECK(S.rows() == (n_coil * n_src));
    CHECK(S.cols() == n_read);
    CHECK((S.array().real() > 0.).all());
    CHECK(S(0, 0).real() == Approx(1. / scale));
    CHECK(
        S((n_coil * n_src) - 1, n_read - 1).real() == Approx((n_readpoints + n_coil - 1) / scale));
  }

  SECTION("Grab Targets Single")
  {
    long const n_read = 2;
    auto const T = GrabTargets(kspace, 2.f, 1, 0, 1, 0, n_read);
    CHECK(T.rows() == n_coil);
    CHECK(T.cols() == n_read);
    CHECK((T.array().real() > 0.).all());
    CHECK(T(0, 0).real() == Approx(1. / scale));
    CHECK(T(n_coil - 1, 0).real() == Approx((n_read - 1 + n_coil - 1) / scale));
  }

  SECTION("Grab Targets Multiple")
  {
    long const n_read = 2;
    long const n_tgt = 2;
    auto const T = GrabTargets(kspace, 2.f, n_tgt, 0, 1, 0, n_read);
    CHECK(T.rows() == n_coil * n_tgt);
    CHECK(T.cols() == n_read);
    CHECK((T.array().real() > 0.).all());
    CHECK(T(0, 0).real() == Approx(1. / scale));
    CHECK(
        T((n_coil * n_tgt) - 1, n_read - 1).real() ==
        Approx((n_read - 1 + n_coil - 1 + n_tgt) / scale));
  }

  SECTION("Fill Targets Iterative")
  {
    long const n_src = 4;
    long const n_tgt = 2;
    Cx3 gap(n_coil, n_tgt, 1);
    gap.setZero();
    Eigen::MatrixXcd S = Eigen::MatrixXcd::Ones(n_coil * n_src, 1);
    Eigen::MatrixXcd W = Eigen::MatrixXcd::Identity(n_coil, n_coil * n_src);
    FillIterative(S, W, 1.f, 0, n_tgt, gap);
    // 0.5 from k0 being halved
    CHECK(R0(gap.abs().sum())() == ((n_tgt - 0.5) * n_coil));
  }

  SECTION("Fill Targets Simultaneous")
  {
    long const n_src = 4;
    long const n_tgt = 3;
    Cx3 gap(n_coil, n_tgt, 1);
    gap.setZero();
    Eigen::MatrixXcd S = Eigen::MatrixXcd::Ones(n_coil * n_src, 1);
    Eigen::MatrixXcd W = Eigen::MatrixXcd::Identity(n_coil * n_tgt, n_coil * n_src);
    FillSimultaneous(S, W, 1.f, 0, gap);
    // 0.5 from k0 being halved
    CHECK(R0(gap.abs().sum())() == ((n_tgt - 0.5) * n_coil));
  }
}

TEST_CASE("ZINFANDEL Algorithm", "[ZINFANDEL]")
{
  Log log(false);
  long const n_read = 12;
  long const n_spoke = 8;
  long const n_coil = 4;
  Cx3 kspace(n_coil, n_read, n_spoke);
  kspace.setZero();

  for (auto is = 0; is < n_spoke; is++) {
    for (auto ir = 0; ir < n_read; ir++) {
      for (auto ic = 0; ic < n_coil; ic++) {
        kspace(ic, ir, is) = std::complex<float>(ir + ic + is + 1, 0);
      }
    }
  }
  // k0 is half a sample
  kspace.slice(Sz3{0, 0, 0}, Sz3{n_coil, 1, n_spoke}) =
      kspace.slice(Sz3{0, 0, 0}, Sz3{n_coil, 1, n_spoke}) / std::complex<float>(2.f, 0.f);

  SECTION("Linear Iterative")
  {
    long const n_gap = 2;
    long const n_src = 4;
    Cx3 test_kspace = kspace;
    test_kspace.slice(Sz3{0, 0, 0}, Sz3{n_coil, n_gap, n_spoke}).setZero();
    zinfandel(n_gap, 1, n_src, 2, 4, 0.0, test_kspace, log);
    Cx3 diff = test_kspace - kspace;
    float const sum_diff = norm(diff) / diff.size();
    CHECK(sum_diff == Approx(0.f).margin(1.e-6f));
  }

  SECTION("Linear Simultaneous")
  {
    long const n_gap = 2;
    long const n_src = 4;
    Cx3 test_kspace = kspace;
    test_kspace.slice(Sz3{0, 0, 0}, Sz4{n_coil, n_gap, n_spoke}).setZero();
    zinfandel(n_gap, n_gap, n_src, 2, 4, 0.0, test_kspace, log);
    Cx3 diff = test_kspace.slice(Sz3{0, 0, 0}, Sz3{n_coil, n_gap, n_spoke}) -
               kspace.slice(Sz3{0, 0, 0}, Sz3{n_coil, n_gap, n_spoke});
    float const sum_diff = norm(diff) / diff.size();
    CHECK(sum_diff == Approx(0.f).margin(1.e-6f));
  }
}