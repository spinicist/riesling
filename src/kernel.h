#pragma once

#include "types.h"
#include <functional>
#include <vector>

struct InterpPair
{
  Size3 point;
  float weight;
};

using ApodizeFunction = std::function<void(Cx3 &img, bool const adjoint)>;

struct Kernel
{
  virtual long radius() const = 0;
  virtual Sz3 start() const = 0;
  virtual Sz3 size() const = 0;
  virtual Cx3 kspace(Point3 const &x) const = 0;                //!< Returns the k-space kernel
  virtual Cx3 image(Point3 const &x, Dims3 const &G) const = 0; //!< Returns the image space kernel
  virtual Cx4 sensitivity(
      Point3 const &x, Cx4 const &s) const = 0; //!< Calculates the sensitivity kernel in k-space
  virtual ApodizeFunction apodization(Dims3 const &gridDims) const = 0;
};
