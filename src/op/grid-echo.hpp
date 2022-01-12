#pragma once

#include "../kernel.h"
#include "../trajectory.h"
#include "gridBase.hpp"
#include "operator.h"
#include <memory>

struct GridOp : GridBase
{
  GridOp(Mapping map, bool const unsafe, Log &log)
    : GridBase(map, unsafe, log)
  {
  }
  virtual ~GridOp(){};

  Sz3 outputDimensions() const
  {
    return mapping_.noncartDims;
  }

  Sz5 inputDimensions(Index const nc) const
  {
    return Sz5{
      nc, mapping_.echoes, mapping_.cartDims[0], mapping_.cartDims[1], mapping_.cartDims[2]};
  }

  Sz5 inputDimensions(Index const nc, Index const ne) const
  {
    return Sz5{nc, ne, mapping_.cartDims[0], mapping_.cartDims[1], mapping_.cartDims[2]};
  }

  virtual void A(Input const &x, Output &y) const = 0;
  virtual void Adj(Output const &x, Input &y) const = 0;

  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid

protected:
};
