#include "interpolator.h"
#include "kaiser-bessel.h"

#include <cfenv>
#include <cmath>

void NearestNeighbor::interpolate(
    Point3 const &gp, std::complex<float> const value, Cx3 &cart) const
{
  assert(cart.dimension(0) == cart.dimension(1));
  assert(cart.dimension(1) == cart.dimension(2));
  long const gridSz = cart.dimension(0);
  std::fesetround(FE_TONEAREST);
  auto const gi = wrap(
      Size3{gp.unaryExpr([](float const &x) { return std::nearbyint(x); }).cast<Eigen::Index>()},
      gridSz);
  cart(gi[0], gi[1], gi[2]) += value;
}

void NearestNeighbor::interpolate(Point3 const &gp, Cx1 const value, Cx4 &cart) const
{
  assert(cart.dimension(0) == cart.dimension(1));
  assert(cart.dimension(1) == cart.dimension(2));
  long const gridSz = cart.dimension(0);
  std::fesetround(FE_TONEAREST);
  auto const gi = wrap(
      Size3{gp.unaryExpr([](float const &x) { return std::nearbyint(x); }).cast<Eigen::Index>()},
      gridSz);
  cart.chip(gi[2], 2).chip(gi[1], 1).chip(gi[0], 0) += value;
}

void NearestNeighbor::deapodize(Dims3 const /* */, Cx3 & /* */) const
{
  // Null-op
}

KaiserBessel::KaiserBessel(float const os)
{
  w_ = os * 2.5;   // Full width
  beta_ = os * w_; // 3 is a magic number from GE code

  // This is only approximate as real kernels are calculated on the fly
  scale_ = 0.f;
  long const hsz = w_ / 2;
  for (long iz = -hsz; iz < hsz; iz++) {
    float const z = KB(beta_, iz, w_);
    for (long iy = -hsz; iy < hsz; iy++) {
      float const y = KB(beta_, iy, w_);
      for (long ix = -hsz; ix < hsz; ix++) {
        float const x = KB(beta_, iz, w_);
        scale_ += x * y * z;
      }
    }
  }
}

void KaiserBessel::interpolate(Point3 const &gp, std::complex<float> const value, Cx3 &cart) const
{
  assert(cart.dimension(0) == cart.dimension(1) == cart.dimension(2));
  long const gridSz = cart.dimension(0);
  std::fesetround(FE_TONEAREST);
  Size3 index = {static_cast<long>(std::nearbyint(gp[0])),
                 static_cast<long>(std::nearbyint(gp[1])),
                 static_cast<long>(std::nearbyint(gp[2]))};
  Point3 const offset = gp - index.cast<float>();
  long const hsz = w_ / 2;
  for (long iz = -hsz; iz < hsz; iz++) {
    float const z = KB(beta_, iz - offset(2), w_);
    for (long iy = -hsz; iy < hsz; iy++) {
      float const y = KB(beta_, iy - offset(1), w_);
      for (long ix = -hsz; ix < hsz; ix++) {
        float const x = KB(beta_, ix - offset(0), w_);
        Size3 pi = wrap(Size3(index + Size3{ix, iy, iz}), gridSz);
        if ((pi >= 0).all() && (pi < cart.dimension(0)).all()) {
          cart(pi[0], pi[1], pi[2]) += value * x * y * z / scale_;
        }
      }
    }
  }
}

void KaiserBessel::interpolate(Point3 const &gp, Cx1 const vals, Cx4 &cart) const
{
  assert(cart.dimension(0) == cart.dimension(1) == cart.dimension(2));
  long const gridSz = cart.dimension(0);
  long const nc = cart.dimension(3);
  std::fesetround(FE_TONEAREST);
  Size3 index = {static_cast<long>(std::nearbyint(gp[0])),
                 static_cast<long>(std::nearbyint(gp[1])),
                 static_cast<long>(std::nearbyint(gp[2]))};
  Point3 const offset = gp - index.cast<float>();
  long const hsz = w_ / 2;
  for (long iz = -hsz; iz < hsz; iz++) {
    float const z = KB(beta_, iz - offset(2), w_);
    for (long iy = -hsz; iy < hsz; iy++) {
      float const y = KB(beta_, iy - offset(1), w_);
      for (long ix = -hsz; ix < hsz; ix++) {
        float const x = KB(beta_, ix - offset(0), w_);
        Size3 pi = wrap(Size3(index + Size3{ix, iy, iz}), gridSz);
        if ((pi >= 0).all() && (pi < cart.dimension(0)).all()) {
          cart.slice(Sz4{pi[0], pi[1], pi[2], 0}, Sz4{1, 1, 1, nc}) +=
              vals * vals.constant(x * y * z / scale_);
        }
      }
    }
  }
}

void KaiserBessel::deapodize(Dims3 const fullSize, Cx3 &image) const
{
  float const scale = KB_FT(beta_, 0., w_);
  auto roc = [&](int const sz, int const ref_sz) {
    Eigen::ArrayXf r(sz);
    for (long ii = 0; ii < sz; ii++) {
      float const pos = (ii - sz / 2.f) / static_cast<float>(ref_sz);
      r(ii) = KB_FT(beta_, pos, w_) / scale;
    }
    return r;
  };
  auto const rocX = roc(image.dimension(0), fullSize[0]);
  auto const rocY = roc(image.dimension(1), fullSize[1]);
  auto const rocZ = roc(image.dimension(2), fullSize[2]);

  for (Eigen::Index iz = 0; iz < image.dimension(2); iz++) {
    float const rz = rocZ(iz);
    for (Eigen::Index iy = 0; iy < image.dimension(1); iy++) {
      float const ry = rocY(iy) * rz;
      for (Eigen::Index ix = 0; ix < image.dimension(0); ix++) {
        float const rx = rocX(ix) * ry;
        image(ix, iy, iz) *= rx;
      }
    }
  }
}