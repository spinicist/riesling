#pragma once

#include "../trajectory.h"
#include "operator.h"

struct GridBase : Operator<5, 3>
{
  GridBase(Mapping map, bool const unsafe, Log &log);
  virtual ~GridBase(){};

  virtual Input::Dimensions inputDimensions(long const nc) const = 0;
  virtual Output::Dimensions outputDimensions() const = 0;

  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid

  Sz3 gridDims() const; // Returns the dimensions of the grid
  void setSDC(float const dc);
  void setSDC(R2 const &sdc);
  void setUnsafe();
  void setSafe();

  Mapping const &mapping() const;
  R2 SDC() const;

protected:
  Mapping mapping_;
  bool safe_;
  Log log_;
};