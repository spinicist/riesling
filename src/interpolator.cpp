#include "fmt/ostream.h"
#include "interpolator.h"
#include "kaiser-bessel.h"
#include <cfenv>
#include <cmath>

Interpolator const *
GetInterpolator(bool const kb, float const os, bool const stack, Dims3 const &grid)
{
  if (kb) {
    return new KaiserBessel(os, stack, grid);
  } else {
    return new NearestNeighbor();
  }
}

long NearestNeighbor::kernelSize() const
{
  return 1;
}

std::vector<KernelPoint> NearestNeighbor::kernel(Point3 const & /* */) const
{
  return {KernelPoint{Point3{0.f, 0.f, 0.f}, 1.f}};
}

void NearestNeighbor::apodize(Cx3 & /* */) const
{
  // Null-op
}

void NearestNeighbor::deapodize(Cx3 & /* */) const
{
  // Null-op
}

KaiserBessel::KaiserBessel(float const os, bool const stack, Dims3 const &full)
{
  w_ = 3; // Full width
  beta_ = os * 2 * w_;

  if (stack) {
    is3D_ = false;
    sz_ = w_ * w_;
  } else {
    is3D_ = true;
    sz_ = w_ * w_ * w_;
  }

  float const scale = KB_FT(0., beta_);
  auto apod = [&](int const sz) {
    Eigen::ArrayXf r(sz);
    for (long ii = 0; ii < sz; ii++) {
      float const pos = (ii - sz / 2.f) / static_cast<float>(sz);
      r(ii) = KB_FT(pos * w_, beta_) / scale;
    }
    return r;
  };
  apodX_ = apod(full[0]);
  apodY_ = apod(full[1]);
  if (is3D_) {
    apodZ_ = apod(full[2]);
  } else {
    apodZ_ = Eigen::ArrayXf::Ones(1);
  }
}

long KaiserBessel::kernelSize() const
{
  return sz_;
}

std::vector<KernelPoint> KaiserBessel::kernel(Point3 const &gp) const
{
  std::fesetround(FE_TONEAREST);
  Point3 index = {std::nearbyint(gp[0]), std::nearbyint(gp[1]), std::nearbyint(gp[2])};
  Point3 const center = gp - index;

  long const hsz = w_ / 2;
  Eigen::ArrayXf const indices = Eigen::ArrayXf::LinSpaced(w_, -hsz, hsz);
  Eigen::ArrayXf const x = indices + center(0);
  Eigen::ArrayXf const y = indices + center(1);
  Eigen::ArrayXf const z = is3D_ ? Eigen::ArrayXf(indices + center(2)) : Eigen::ArrayXf::Zero(1);

  Eigen::ArrayXf const b_x = KB(x / w_, beta_);
  Eigen::ArrayXf const b_y = KB(y / w_, beta_);
  Eigen::ArrayXf const b_z = is3D_ ? KB(z / w_, beta_) : Eigen::ArrayXf::Ones(1);

  // fmt::print("bx {} by {} bz {}\n", b_x.transpose(), b_y.transpose(), b_z.transpose());

  float total = 0.f;
  std::vector<KernelPoint> k(sz_);
  long ik = 0;
  for (long iz = 0; iz < (is3D_ ? w_ : 1); iz++) {
    for (long iy = 0; iy < w_; iy++) {
      for (long ix = 0; ix < w_; ix++) {
        Point3 const offset = Point3(x[ix], y[iy], z[iz]);
        float val = b_x[ix] * b_y[iy] * b_z[iz];
        total += val;
        k[ik++] = KernelPoint{.offset = offset, .weight = val};
      }
    }
  }
  std::transform(k.begin(), k.end(), k.begin(), [total](KernelPoint const &kp) {
    return KernelPoint{kp.offset, kp.weight / total};
  });

  return k;
}

void KaiserBessel::apodize(Cx3 &image) const
{
  long const stz = (apodZ_.rows() - image.dimension(2)) / 2;
  long const sty = (apodY_.rows() - image.dimension(1)) / 2;
  long const stx = (apodX_.rows() - image.dimension(0)) / 2;
  for (Eigen::Index iz = 0; iz < image.dimension(2); iz++) {
    float const rz = apodZ_(stz + iz);
    for (Eigen::Index iy = 0; iy < image.dimension(1); iy++) {
      float const ry = apodY_(sty + iy) * rz;
      for (Eigen::Index ix = 0; ix < image.dimension(0); ix++) {
        float const rx = apodX_(stx + ix) * ry;
        image(ix, iy, iz) /= rx;
      }
    }
  }
}

void KaiserBessel::deapodize(Cx3 &image) const
{
  long const stz = (apodZ_.rows() - image.dimension(2)) / 2;
  long const sty = (apodY_.rows() - image.dimension(1)) / 2;
  long const stx = (apodX_.rows() - image.dimension(0)) / 2;
  for (Eigen::Index iz = 0; iz < image.dimension(2); iz++) {
    float const rz = apodZ_(stz + iz);
    for (Eigen::Index iy = 0; iy < image.dimension(1); iy++) {
      float const ry = apodY_(sty + iy) * rz;
      for (Eigen::Index ix = 0; ix < image.dimension(0); ix++) {
        float const rx = apodX_(stx + ix) * ry;
        image(ix, iy, iz) *= rx;
      }
    }
  }
}