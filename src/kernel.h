#pragma once

#include "types.h"

struct InterpPair
{
  Size3 point;
  float weight;
};

struct Kernel
{
  virtual float radius() const = 0;
  virtual Sz3 start() const = 0;
  virtual Sz3 size() const = 0;
  virtual R3 kspace(Point3 const &x) const = 0;                 //!< Returns the k-space kernel
  virtual Cx3 image(Point3 const &x, Dims3 const &G) const = 0; //!< Returns the image space kernel
  virtual void sqrtOn(){};  //!< Defined this way so NN doesn't have to do anything
  virtual void sqrtOff(){}; //!< Defined this way so NN doesn't have to do anything
};
