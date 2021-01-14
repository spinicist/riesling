#pragma once

#include "kernel.h"

struct KaiserBessel final : Kernel
{
  KaiserBessel(long const w, float const os, bool const threeD = true);
  long radius() const;
  Sz3 start() const;
  Sz3 size() const;
  Cx3 kspace(Point3 const &x) const;
  Cx3 image(Point3 const &x, Dims3 const &G) const;
  Cx4 sensitivity(Point3 const &x, Cx4 const &s) const;
  ApodizeFunction apodization(Dims3 const &dims) const;

private:
  long w_;
  float beta_;
  bool threeD_;

  Sz3 st_, sz_;
  R1 p_;
  R1 lookup_;
};
