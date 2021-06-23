#pragma once

#include "kernel.h"

struct NearestNeighbour final : Kernel
{
  NearestNeighbour(long const w = 1);
  long radius() const;
  Sz3 start() const;
  Sz3 size() const;
  R3 kspace(Point3 const &x) const;
  Cx3 image(Point3 const &x, Sz3 const &G) const;

private:
  long w_;
};
