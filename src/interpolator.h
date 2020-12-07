#pragma once
#include "types.h"

struct KernelPoint
{
  Point3 offset;
  float weight;
};

/** Interpolator interface. Declaring this using inheritance is slight overkill as there are only
 * two interpolators (nearest-neighbor and Kaiser-Bessel) but leaves space for more in future
 */
struct Interpolator
{
  virtual long kernelSize() const = 0;
  virtual std::vector<KernelPoint> kernel(Point3 const &p) const = 0;
  virtual void apodize(Cx3 &image) const = 0;
  virtual void deapodize(Cx3 &image) const = 0;
};

struct NearestNeighbor final : Interpolator
{
  virtual long kernelSize() const;
  virtual std::vector<KernelPoint> kernel(Point3 const &p) const;
  virtual void apodize(Cx3 &image) const;
  virtual void deapodize(Cx3 &image) const;
};

struct KaiserBessel final : Interpolator
{
  KaiserBessel(float const os, bool const stack, Dims3 const &full);
  ~KaiserBessel() = default;

  virtual long kernelSize() const;
  virtual std::vector<KernelPoint> kernel(Point3 const &p) const;
  virtual void apodize(Cx3 &image) const;
  virtual void deapodize(Cx3 &image) const;

private:
  bool is3D_;
  float beta_;
  long w_, sz_;
  Eigen::ArrayXf apodX_, apodY_, apodZ_;
};

/**
 * @brief Factory function to return a useful Interpolator
 *
 * @param kb    Use Kaiser-Bessel interpolation
 * @param os    Oversampling
 * @param stack Is a stack-of-stars type recon
 * @param grid  Grid size
 * @return Interpolator const*
 */
Interpolator const *
GetInterpolator(bool const kb, float const os, bool const stack, Dims3 const &grid);