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
  Eigen::VectorXcf z = x;
  prox.apply(1.f, z);
  INFO("x " << x.transpose() << "\nz " << z.transpose() << "\ny " << y.transpose());
  CHECK((y - z).norm() == Approx(0.f).margin(1.e-6f));

  x = RandN(sz, 1.f);
  z = x;
  prox.conj(1.f, z);
  Eigen::ArrayXf y2 = (x.array().abs() > 1.f).select(1.f, x.array().abs());
  INFO("x " << x.array().abs().transpose() << "\nz " << z.array().abs().transpose() << "\ny2 " << y2.transpose());
  CHECK((y2 - z.array().abs()).matrix().norm() == Approx(0.f).margin(1.e-6f));

  x = RandN(sz, 1.e-3f);
  z = x;
  prox.conj(1.f, z);
  y2 = (x.array().abs() > 1.f).select(1.f, x.array().abs());
  INFO("x " << x.array().abs().transpose() << "\nz " << z.array().abs().transpose() << "\ny2 " << y2.transpose());
  CHECK((y2 - z.array().abs()).matrix().norm() == Approx(0.f).margin(1.e-6f));

  x = RandN(sz, 10.f);
  z = x;
  prox.conj(1.f, z);
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
  Eigen::VectorXcf z = x;
  prox.apply(0.1f, z);
  INFO("x " << x.transpose() << "\nz " << z.transpose() << "\nxr\n"
            << (x * (1.f - λ * 0.1f * std::sqrt(sz) / x.norm())).transpose() << "\n");
  CHECK((z - (x * (1.f - λ * 0.1f / x.norm()))).norm() == Approx(0.f).margin(1.e-6f));

  x = RandN(sz, 10.f);
  z = x;
  prox.conj(0.1f, z);
  INFO("x " << x.transpose() << "\nz " << z.transpose() << "\nxr\n" << (x.array() * λ / x.norm()).transpose() << "\n");
  CHECK((z.array() - (x.array() * λ / x.norm())).matrix().norm() == Approx(0.f).margin(1.e-6f));
}

TEST_CASE("ConjL1", "[prox]")
{
  Index const sz = 6;
  float const λ = 0.25f;
  float const α = 0.5f;
  auto        l1 = rl::Proxs::L1::Make(λ, sz);
  auto        l1c = rl::Proxs::Conjugate::Make(l1);

  Eigen::VectorXcf x = RandN(sz, 1.f);
  INFO("x  " << x.transpose() << "\n");

  Eigen::VectorXcf za = x;
  Eigen::VectorXcf cc = x;

  l1->apply(α, za);
  l1c->conj(α, cc);

  INFO("za " << za.transpose() << "\ncc " << cc.transpose() << "\n");
  CHECK((za - cc).norm() == Approx(0.f).margin(1.e-6f));

  Eigen::VectorXcf zc = x;
  Eigen::VectorXcf ca = x;

  l1->conj(α, zc);
  l1c->apply(α, ca);
  INFO("zc " << zc.transpose() << "\nca " << ca.transpose() << "\n");
  CHECK((zc - ca).norm() == Approx(0.f).margin(1.e-6f));
}

TEST_CASE("ConjL2", "[prox]")
{
  Index const sz = 6;
  float const λ = 0.25f;
  float const α = 0.5f;
  auto        l2 = rl::Proxs::L2<1, 1>::Make(λ, Sz1{sz}, Sz1{0});
  auto        l2c = rl::Proxs::Conjugate::Make(l2);

  Eigen::VectorXcf x = RandN(sz, 1.f);
  INFO("x  " << x.transpose() << "\n");

  Eigen::VectorXcf za = x;
  Eigen::VectorXcf cc = x;
  l2->apply(α, za);
  l2c->conj(α, cc);

  INFO("za " << za.transpose() << "\ncc " << cc.transpose() << "\n");
  CHECK((za - cc).norm() == Approx(0.f).margin(1.e-6f));

  Eigen::VectorXcf zc = x;
  Eigen::VectorXcf ca = x;

  l2->conj(α, zc);
  l2c->apply(α, ca);
  INFO("zc " << zc.transpose() << "\nca " << ca.transpose() << "\n");
  CHECK((zc - ca).norm() == Approx(0.f).margin(1.e-6f));
}

TEST_CASE("ConjSoS", "[prox]")
{
  Index const            sz = 6;
  Eigen::VectorXcf const x = RandN(sz, 1.f), b = RandN(sz, 1.f);
  float const            α = 0.5f;
  auto                   res = rl::Proxs::SumOfSquares::Make(rl::Proxs::Prox::CMap(b.data(), b.size()));
  auto                   rc = rl::Proxs::Conjugate::Make(res);

  INFO("b  " << b.transpose() << "\n");
  INFO("x  " << x.transpose() << "\n");

  Eigen::VectorXcf za = x;
  Eigen::VectorXcf cc = x;
  res->apply(α, za);
  rc->conj(α, cc);

  INFO("za " << za.transpose() << "\ncc " << cc.transpose() << "\n");
  CHECK((za - cc).norm() == Approx(0.f).margin(1.e-6f));

  Eigen::VectorXcf zc = x;
  Eigen::VectorXcf ca = x;
  res->conj(α, zc);
  rc->apply(α, ca);
  INFO("zc " << zc.transpose() << "\nca " << ca.transpose() << "\n");
  CHECK((zc - ca).norm() == Approx(0.f).margin(1.e-6f));
}