#pragma once

#include "kernel.h"

struct KaiserBessel final : Kernel
{
  KaiserBessel(long const w, float const os, bool const threeD = true);
  float radius() const;
  Sz3 start() const;
  Sz3 size() const;
  R3 kspace(Point3 const &x) const;
  Cx3 image(Point3 const &x, Dims3 const &G) const;

private:
  long w_;
  float beta_;
  bool threeD_;
  Sz3 st_, sz_;
  R1 p_;
};
