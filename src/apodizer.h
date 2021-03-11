#pragma once

#include "kernel.h"
#include "log.h"
#include "types.h"

/*! There are analytic expressions for the image space apodization of most
 *  kernels. However, given the subtle differences between e.g. separable and
 *  spherically symmetric kernels, for the time being we take a brute force
 *  approach - embed the kernel in the correct size grid and do an FFT
 */
struct Apodizer
{
  Apodizer(Kernel *const k, Dims3 const &grid, Dims3 const &img, Log &log);

  void apodize(Cx3 &x);
  void deapodize(Cx3 &x);

private:
  Log &log_;
  R3 y_;
};
