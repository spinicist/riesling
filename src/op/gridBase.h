#pragma once

#include "../trajectory.h"

struct GridBase
{
  GridBase(Mapping map, bool const unsafe, Log &log);
  virtual ~GridBase(){};

  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid

  Sz3 gridDims() const; // Returns the dimensions of the grid
  void setSDCExponent(float const dce);
  void setSDC(float const dc);
  void setSDC(R2 const &sdc);
  void setUnsafe();
  void setSafe();
  void sqrtOn(); // Use square-root of gridding kernel for Pipe SDC
  void sqrtOff();

  Mapping const &mapping() const;
  R2 SDC() const;

protected:
  Mapping mapping_;
  bool safe_, sqrt_;
  Log log_;
  float DCexp_;
};