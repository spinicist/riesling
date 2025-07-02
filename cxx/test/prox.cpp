#include "rl/log/log.hpp"
#include "rl/prox/norms.hpp"

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

using namespace rl;
using namespace Catch;

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
  Eigen::VectorXcf z(sz);
  prox.primal(1.f, x, z);
  INFO("x " << x.transpose() << "\nz " << z.transpose() << "\ny " << y.transpose());
  CHECK((y - z).norm() == Approx(0.f).margin(1.e-6f));

  // Simple Box Muller transform
  Eigen::ArrayXf u = (Eigen::ArrayXf::Random(sz) * 0.5f) + 0.5f;
  Eigen::ArrayXf v = (Eigen::ArrayXf::Random(sz) * 0.5f) + 0.5f;
  x.real() = (-2. * u.log()).sqrt() * cos(2. * M_PI * v);
  x.imag() = (-2. * v.log()).sqrt() * sin(2. * M_PI * u);
  prox.dual(1.f, x, z);
  Eigen::ArrayXf y2 = (x.array().abs() > 1.f).select(1.f, x.array().abs());
  INFO("x " << x.array().abs().transpose() << "\nz " << z.array().abs().transpose() << "\ny2 " << y2.transpose());
  CHECK((y2 - z.array().abs()).matrix().norm() == Approx(0.f).margin(1.e-6f));

  u = (Eigen::ArrayXf::Random(sz) * 0.5f) + 0.5f;
  v = (Eigen::ArrayXf::Random(sz) * 0.5f) + 0.5f;
  x.real() = 0.001 * (-2. * u.log()).sqrt() * cos(2. * M_PI * v);
  x.imag() = 0.001 * (-2. * v.log()).sqrt() * sin(2. * M_PI * u);
  prox.dual(1.f, x, z);
  y2 = (x.array().abs() > 1.f).select(1.f, x.array().abs());
  INFO("x " << x.array().abs().transpose() << "\nz " << z.array().abs().transpose() << "\ny2 " << y2.transpose());
  CHECK((y2 - z.array().abs()).matrix().norm() == Approx(0.f).margin(1.e-6f));

  u = (Eigen::ArrayXf::Random(sz) * 0.5f) + 0.5f;
  v = (Eigen::ArrayXf::Random(sz) * 0.5f) + 0.5f;
  x.real() = 10 * (-2. * u.log()).sqrt() * cos(2. * M_PI * v);
  x.imag() = 10 * (-2. * v.log()).sqrt() * sin(2. * M_PI * u);
  prox.dual(1.f, x, z);
  y2 = (x.array().abs() > 1.f).select(1.f, x.array().abs());
  INFO("x " << x.array().abs().transpose() << "\nz " << z.array().abs().transpose() << "\ny2 " << y2.transpose());
  CHECK((y2 - z.array().abs()).matrix().norm() == Approx(0.f).margin(1.e-6f));
}
