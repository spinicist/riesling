#pragma once

#include "../kernel.h"
#include "../trajectory.h"
#include "gridBase.hpp"
#include "operator.h"
#include <memory>

struct GridBasisOp : GridBase
{
  GridBasisOp(Mapping map, bool const unsafe, R2 basis, Log &log)
    : GridBase(map, unsafe, log)
    , basis_{basis}
    , basisScale_{std::sqrt((float)basis_.dimension(0))}
  {
    log_.info("Basis size {}x{}, scale {}", basis_.dimension(0), basis_.dimension(1), basisScale_);
  }

  Index dimension(Index const D) const
  {
    assert(D < 3);
    return mapping_.cartDims[D];
  }

  Sz5 inputDimensions(Index const nc) const
  {
    return Sz5{
      nc, basis_.dimension(1), mapping_.cartDims[0], mapping_.cartDims[1], mapping_.cartDims[2]};
  }

  Sz3 outputDimensions() const
  {
    return mapping_.noncartDims;
  }

  R2 const &basis() const
  {
    return basis_;
  }

  virtual ~GridBasisOp(){};

  virtual void A(Input const &x, Output &y) const = 0;
  virtual void Adj(Output const &x, Input &y) const = 0;

protected:
  R2 basis_;
  float basisScale_;
};
