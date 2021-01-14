#include "kernel_kb.h"

#include "tensorOps.h"
#include "threads.h"
#include <fmt/ostream.h>

float KB_FT(float const &x, float const &beta)
{
  double const a = sqrt(pow(beta, 2.) - pow(M_PI * x, 2.));
  return a / sinh(a); // * bessel_i0(beta);
}

KaiserBessel::KaiserBessel(long const w, float const os, bool const threeD)
    : w_{w}
    , beta_{(float)M_PI * sqrtf(pow(w_ * (os - 0.5f) / os, 2.f) - 0.8f)}
    , threeD_{threeD}
{
  std::fill(sz_.begin(), sz_.end(), w_);
  if (!threeD) {
    sz_[2] = 1;
  }
  std::transform(sz_.begin(), sz_.end(), st_.begin(), [](long const d) { return -d / 2; });

  p_.resize(w_);
  std::iota(p_.data(), p_.data() + p_.size(), -w / 2);

  // Build lookup table
  R1 lx(512);
  std::iota(lx.data(), lx.data() + lx.size(), 0);
  lx = lx * lx.constant(w_ / (2.f * (lx.size() - 1)));
  lookup_ = (lx.constant(beta_) * (lx.constant(1.f) - (lx * lx.constant(2.f / w)).square()).sqrt())
                .bessel_i0();
}

ApodizeFunction KaiserBessel::apodization(Dims3 const &dims) const
{
  float const scale = KB_FT(0., beta_);
  auto apod = [&](int const sz) {
    R1 r(sz);
    for (long ii = 0; ii < sz; ii++) {
      float const pos = (ii - sz / 2.f) / static_cast<float>(sz);
      r(ii) = KB_FT(pos * w_, beta_) / scale;
    }
    return r;
  };
  R1 apodX = apod(dims[0]);
  R1 apodY = apod(dims[1]);
  R1 apodZ;
  if (threeD_) {
    apodZ = apod(dims[2]);
  } else {
    apodZ = R1(dims[2]);
    apodZ.setConstant(1.f);
  }

  ApodizeFunction a = [apodX, apodY, apodZ](Cx3 &image, bool const adjoint) {
    long const sz_z = image.dimension(2);
    long const sz_y = image.dimension(1);
    long const sz_x = image.dimension(0);
    long const st_z = (apodZ.size() - sz_z) / 2;
    long const st_y = (apodY.size() - sz_y) / 2;
    long const st_x = (apodX.size() - sz_x) / 2;

    auto const full =
        (apodZ.slice(Sz1{st_z}, Sz1{sz_z}).reshape(Sz3{1, 1, sz_z}).broadcast(Sz3{sz_x, sz_y, 1}) *
         apodY.slice(Sz1{st_y}, Sz1{sz_y}).reshape(Sz3{1, sz_y, 1}).broadcast(Sz3{sz_x, 1, sz_z}) *
         apodX.slice(Sz1{st_x}, Sz1{sz_x}).reshape(Sz3{sz_x, 1, 1}).broadcast(Sz3{1, sz_y, sz_z}))
            .cast<std::complex<float>>();

    if (adjoint) {
      image.device(Threads::GlobalDevice()) = image * full;
    } else {
      image.device(Threads::GlobalDevice()) = image / full;
    }
  };
  return a;
}

long KaiserBessel::radius() const
{
  return w_ / 2;
}

Sz3 KaiserBessel::start() const
{
  return st_;
}

Sz3 KaiserBessel::size() const
{
  return sz_;
}

Cx3 KaiserBessel::kspace(Point3 const &r) const
{
  auto kbLookup = [this](float const x) {
    float const xa = std::abs(x);
    long const sz = lookup_.size();
    if (xa > (w_ / 2)) {
      return 0.f;
    } else {
      float const l = xa * 2.f * (sz - 1.f) / w_;
      long const lo = std::floor(l);
      long const hi = std::ceil(l);
      long const t = l - lo;
      return lookup_(lo) + t * (lookup_(hi) - lookup_(lo));
    }
  };

  R1 const kx = (p_.constant(r[0]) - p_).unaryExpr(kbLookup);
  R1 const ky = (p_.constant(r[1]) - p_).unaryExpr(kbLookup);
  R1 kz;
  if (threeD_) {
    kz = (p_.constant(r[2]) - p_).unaryExpr(kbLookup);
  } else {
    kz.resize(Sz1{1});
    kz.setConstant(1.f);
  }

  R3 k = Outer(Outer(kx, ky), kz);
  k = k / Sum(k);

  return k.cast<Cx>();
}

inline decltype(auto) Sinc(Cx1 const &x)
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

Cx4 KaiserBessel::sensitivity(Point3 const &x, Cx4 const &in) const
{
  Cx3 const k = image(x, Dims3{in.dimension(1), in.dimension(2), in.dimension(3)});
  Cx4 s = in * Tile(k, in.dimension(0));
  return s;
}
