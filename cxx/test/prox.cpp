#include "rl/log/log.hpp"
#include "rl/prox/norms.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

auto RandN(Index const sz, float sigma) -> Eigen::ArrayXcf
{
  Eigen::ArrayXf  u = (Eigen::ArrayXf::Random(sz) * 0.5f) + 0.5f;
  Eigen::ArrayXf  v = (Eigen::ArrayXf::Random(sz) * 0.5f) + 0.5f;
  Eigen::ArrayXcf x(sz);
  x.real() = sigma * (-2. * u.log()).sqrt() * cos(2. * M_PI * v);
  x.imag() = sigma * (-2. * v.log()).sqrt() * sin(2. * M_PI * u);
  return x;
}

TEST_CASE("L1Prox", "[prox]")
{
  Index const           sz = 6;
  float const           λ = 1.f;
  rl::Proxs::L1         prox(λ, sz);
  Eigen::ArrayXf const  ph = Eigen::ArrayXf::Random(sz);
  Eigen::ArrayXcf const eph = (ph.cast<Cx>() * Cx(0., M_PI)).exp();
  Eigen::ArrayXf        mag_x(sz), mag_y(sz);
  mag_x << -3.f, -2.f, -1.f, 0.f, 1.f, 2.f;
  mag_y << -2.f, -1.f, 0.f, 0.f, 0.f, 1.f;
  Eigen::VectorXcf x = mag_x.cast<Cx>() * eph;
  Eigen::VectorXcf y = mag_y.cast<Cx>() * eph;
  auto z = prox.apply(1.f, x);
  INFO("x " << x.transpose() << "\nz " << z.transpose() << "\ny " << y.transpose());
  CHECK((y - z).norm() == Approx(0.f).margin(1.e-6f));

  // Simple Box Muller transform
  x = RandN(sz, 1.f);
  z = prox.conj(1.f, x);
  Eigen::ArrayXf y2 = (x.array().abs() > 1.f).select(1.f, x.array().abs());
  INFO("x " << x.array().abs().transpose() << "\nz " << z.array().abs().transpose() << "\ny2 " << y2.transpose());
  CHECK((y2 - z.array().abs()).matrix().norm() == Approx(0.f).margin(1.e-6f));

  x = RandN(sz, 1.e-3f);
  z = prox.conj(1.f, x);
  y2 = (x.array().abs() > 1.f).select(1.f, x.array().abs());
  INFO("x " << x.array().abs().transpose() << "\nz " << z.array().abs().transpose() << "\ny2 " << y2.transpose());
  CHECK((y2 - z.array().abs()).matrix().norm() == Approx(0.f).margin(1.e-6f));

  x = RandN(sz, 10.f);
  z = prox.conj(1.f, x);
  y2 = (x.array().abs() > 1.f).select(1.f, x.array().abs());
  INFO("x " << x.array().abs().transpose() << "\nz " << z.array().abs().transpose() << "\ny2 " << y2.transpose());
  CHECK((y2 - z.array().abs()).matrix().norm() == Approx(0.f).margin(1.e-6f));
}

TEST_CASE("L2Prox", "[prox]")
{
  Index const      sz = 6;
  float const      λ = 1.f;
  rl::Proxs::L2    prox(λ, Sz1{sz}, Sz1{0});
  Eigen::VectorXcf x = RandN(sz, 1.f);
  auto z = prox.apply(0.1f, x);
  INFO("x " << x.transpose() << "\nz " << z.transpose() << "\nxr\n" << (x * (1.f - λ * 0.1f * std::sqrt(sz) / x.norm())).transpose() << "\n");
CHECK((z - (x * (1.f - λ * 0.1f / x.norm()))).norm() == Approx(0.f).margin(1.e-6f));

  x = RandN(sz, 10.f);
  z.setZero();
  z = prox.conj(0.1f, x);
  float const t = λ * 0.1f;
  INFO("x " << x.transpose() << "\nz " << z.transpose() << "\nxr\n" << (x.array() * t / x.norm()).transpose() << "\n");
  CHECK((z.array() - (x.array() * t / x.norm())).matrix().norm() == Approx(0.f).margin(1.e-6f));
}