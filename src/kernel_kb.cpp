#include "kernel_kb.h"

#include "fft1.h"
#include "tensorOps.h"
#include "threads.h"
#include <fmt/ostream.h>

KaiserBessel::KaiserBessel(long const w, float const os, bool const threeD)
    : w_{w}
    , beta_{(float)M_PI * sqrtf(pow(w_ * (os - 0.5f) / os, 2.f) - 0.8f)}
    , threeD_{threeD}
{
  std::fill(sz_.begin(), sz_.end(), w_);
  if (!threeD) {
    sz_[2] = 1;
  }
  std::transform(sz_.begin(), sz_.end(), st_.begin(), [](long const d) { return -(d - 1) / 2; });

  // Array of indices used when building the kernel
  p_.resize(w_);
  std::iota(p_.data(), p_.data() + p_.size(), -w / 2);
}

float KaiserBessel::radius() const
{
  return w_ / 2.f;
}

Sz3 KaiserBessel::start() const
{
  return st_;
}

Sz3 KaiserBessel::size() const
{
  return sz_;
}

template <typename T>
inline decltype(auto) kb(T const &x, float const w, float const beta)
{
  return (x > (w / 2.f))
      .select(
          x.constant(0.f),
          (x.constant(beta) * (x.constant(1.f) - (x * x.constant(2.f / w)).square()).sqrt())
              .bessel_i0());
}

R3 KaiserBessel::kspace(Point3 const &r) const
{
  R1 const kx = kb(p_.constant(r[0]) - p_, w_, beta_);
  R1 const ky = kb(p_.constant(r[1]) - p_, w_, beta_);
  R1 kz;
  if (threeD_) {
    kz = kb(p_.constant(r[2]) - p_, w_, beta_);
  } else {
    kz.resize(Sz1{1});
    kz.setConstant(1.f);
  }

  R3 const k = Outer(Outer(kx, ky), kz);
  return k / Sum(k);
}

template <typename T>
inline decltype(auto) Sinc(T const &x)
{
  return x.unaryExpr([](Cx const xx) { return std::sin(xx) / xx; });
}

Cx3 KaiserBessel::image(Point3 const &r, Dims3 const &G) const
{
  Cx1 const ix = Sinc((((p_.constant(r[0]) - p_) * p_.constant(M_PI * w_ / G[0])).square() -
                       p_.constant(beta_ * beta_))
                          .cast<Cx>()
                          .sqrt());
  Cx1 const iy = Sinc((((p_.constant(r[1]) - p_) * p_.constant(M_PI * w_ / G[1])).square() -
                       p_.constant(beta_ * beta_))
                          .cast<Cx>()
                          .sqrt());
  Cx1 iz;
  if (threeD_) {
    iz = Sinc((((p_.constant(r[2]) - p_) * p_.constant(M_PI * w_ / G[2])).square() -
               p_.constant(beta_ * beta_))
                  .cast<Cx>()
                  .sqrt());
  } else {
    iz.resize(Sz1{1});
    iz.setConstant(1.f);
  }

  Cx3 v = Outer(Outer(ix, iy), iz);
  v = v / Sum(v);

  return v;
}
