#pragma once
#include "types.h"

struct Interpolator
{
  virtual void interpolate(Point3 const &gp, std::complex<float> value, Cx3 &cart) const = 0;
  virtual void interpolate(Point3 const &gp, Cx1 const vals, Cx4 &cart) const = 0;
  virtual void deapodize(Dims3 const fullSize, Cx3 &image) const = 0;
};

struct NearestNeighbor final : Interpolator
{
  virtual void interpolate(Point3 const &gp, std::complex<float> value, Cx3 &cart) const;
  virtual void interpolate(Point3 const &gp, Cx1 const vals, Cx4 &cart) const;
  virtual void deapodize(Dims3 const fullSize, Cx3 &image) const;
};

struct KaiserBessel final : Interpolator
{
  KaiserBessel(float const os);
  ~KaiserBessel() = default;
  virtual void interpolate(Point3 const &gp, std::complex<float> value, Cx3 &cart) const;
  virtual void interpolate(Point3 const &gp, Cx1 const vals, Cx4 &cart) const;
  virtual void deapodize(Dims3 const fullSize, Cx3 &image) const;

private:
  float beta_, scale_;
  long w_;
};