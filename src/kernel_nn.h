#pragma once

#include "kernel.h"

struct NearestNeighbour final : Kernel
{
  NearestNeighbour(long const w = 1);
  float radius() const;
  Sz3 start() const;
  Sz3 size() const;
  float value(Point3 const &x) const;
  R3 kspace(Point3 const &x) const;
  Cx3 image(Point3 const &x, Dims3 const &G) const;
  Cx4 sensitivity(Point3 const &x, Cx4 const &s) const;
  ApodizeFunction apodization(Dims3 const &gridDims) const;

private:
  long w_;
};
