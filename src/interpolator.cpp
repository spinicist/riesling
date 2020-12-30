#include "interpolator.h"
#include "kaiser-bessel.h"
#include "threads.h"

long NearestNeighbour::size() const
{
  return 1;
}

std::vector<InterpPair> NearestNeighbour::weights(Point3 const offset) const
{
  return {InterpPair{.point = {0, 0, 0}, .weight = 1.f}};
}

void NearestNeighbour::apodize(Cx3 &img) const
{ /* Null op */
}

void NearestNeighbour::deapodize(Cx3 &img) const
{ /* Null op */
}

KaiserBessel::KaiserBessel(long const w, float const os, Dims3 const dims, bool const threeD)
    : w_{w}
    , beta_{(float)M_PI * sqrtf(pow(w_ * (os - 0.5f) / os, 2.f) - 0.8f)}
{
  sz_ = threeD ? w_ * w_ * w_ : w_ * w_;
  std::vector<Size3> points;
  points_.reserve(sz_);
  long const lo_xy = -w_ / 2;
  long const hi_xy = w_ / 2 + 1;
  long const lo_z = threeD ? lo_xy : 0;
  long const hi_z = threeD ? hi_xy : 1;
  for (long iz = lo_z; iz < hi_z; iz++) {
    for (long iy = lo_xy; iy < hi_xy; iy++) {
      for (long ix = lo_xy; ix < hi_xy; ix++) {
        points_.push_back({ix, iy, iz});
      }
    }
  }

  float const scale = KB_FT(0., beta_);
  auto apod = [&](int const sz) {
    R1 r(sz);
    for (long ii = 0; ii < sz; ii++) {
      float const pos = (ii - sz / 2.f) / static_cast<float>(sz);
      r(ii) = KB_FT(pos * w_, beta_) / scale;
    }
    return r;
  };
  apodX_ = apod(dims[0]);
  apodY_ = apod(dims[1]);
  if (threeD) {
    apodZ_ = apod(dims[2]);
  } else {
    apodZ_ = R1(dims[2]);
    apodZ_.setConstant(1.f);
  }
}

long KaiserBessel::size() const
{
  return sz_;
}

std::vector<InterpPair> KaiserBessel::weights(Point3 const xyz) const
{
  std::vector<InterpPair> pts;
  pts.reserve(sz_);
  float sum = 0.f;
  for (auto &p : points_) {
    float const r = (xyz - p.cast<float>().matrix()).norm();
    pts.push_back(InterpPair{.point = p, .weight = KB(r / w_, beta_)});
    sum += pts.back().weight;
  }
  for (auto &p : pts) {
    p.weight /= sum;
  }
  return pts;
}

void KaiserBessel::apodize(Cx3 &image) const
{
  long const sz_z = image.dimension(2);
  long const sz_y = image.dimension(1);
  long const sz_x = image.dimension(0);
  long const st_z = (apodZ_.size() - sz_z) / 2;
  long const st_y = (apodY_.size() - sz_y) / 2;
  long const st_x = (apodX_.size() - sz_x) / 2;

  image.device(Threads::GlobalDevice()) =
      image /
      (apodZ_.slice(Sz1{st_z}, Sz1{sz_z}).reshape(Sz3{1, 1, sz_z}).broadcast(Sz3{sz_x, sz_y, 1}) *
       apodY_.slice(Sz1{st_y}, Sz1{sz_y}).reshape(Sz3{1, sz_y, 1}).broadcast(Sz3{sz_x, 1, sz_z}) *
       apodX_.slice(Sz1{st_x}, Sz1{sz_x}).reshape(Sz3{sz_x, 1, 1}).broadcast(Sz3{1, sz_y, sz_z}))
          .cast<std::complex<float>>();
}

void KaiserBessel::deapodize(Cx3 &image) const
{
  long const sz_z = image.dimension(2);
  long const sz_y = image.dimension(1);
  long const sz_x = image.dimension(0);
  long const st_z = (apodZ_.size() - sz_z) / 2;
  long const st_y = (apodY_.size() - sz_y) / 2;
  long const st_x = (apodX_.size() - sz_x) / 2;

  image.device(Threads::GlobalDevice()) =
      image *
      (apodZ_.slice(Sz1{st_z}, Sz1{sz_z}).reshape(Sz3{1, 1, sz_z}).broadcast(Sz3{sz_x, sz_y, 1}) *
       apodY_.slice(Sz1{st_y}, Sz1{sz_y}).reshape(Sz3{1, sz_y, 1}).broadcast(Sz3{sz_x, 1, sz_z}) *
       apodX_.slice(Sz1{st_x}, Sz1{sz_x}).reshape(Sz3{sz_x, 1, 1}).broadcast(Sz3{1, sz_y, sz_z}))
          .cast<std::complex<float>>();
}