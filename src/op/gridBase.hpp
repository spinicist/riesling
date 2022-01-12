#pragma once

#include "../trajectory.h"
#include "operator.h"

struct GridBase : Operator<5, 3>
{
  GridBase(Mapping map, bool const unsafe, Log &log)
    : mapping_{std::move(map)}
    , safe_{!unsafe}
    , weightEchoes_{true}
    , sdcPow_{1.f}
    , log_{log}
  {
  }

  virtual ~GridBase(){};
  virtual Input::Dimensions inputDimensions(Index const nc) const = 0;
  virtual Output::Dimensions outputDimensions() const = 0;
  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid

  void setSDC(float const d)
  {
    std::fill(mapping_.sdc.begin(), mapping_.sdc.end(), d);
  }

  void setSDC(R2 const &sdc)
  {
    std::transform(
      mapping_.noncart.begin(),
      mapping_.noncart.end(),
      mapping_.sdc.begin(),
      [&sdc](NoncartesianIndex const &nc) { return sdc(nc.read, nc.spoke); });
  }

  R2 SDC() const
  {
    R2 sdc(mapping_.noncartDims[1], mapping_.noncartDims[2]);
    sdc.setZero();
    for (size_t ii = 0; ii < mapping_.noncart.size(); ii++) {
      sdc(mapping_.noncart[ii].read, mapping_.noncart[ii].spoke) = mapping_.sdc[ii];
    }
    return sdc;
  }

  void setSDCPower(float const p)
  {
    sdcPow_ = p;
  }

  void setUnsafe()
  {
    safe_ = true;
  }

  void setSafe()
  {
    safe_ = false;
  }

  void doNotWeightEchoes()
  {
    weightEchoes_ = false;
  }

  Mapping const &mapping() const
  {
    return mapping_;
  }

protected:
  Mapping mapping_;
  bool safe_, weightEchoes_;
  float sdcPow_;
  Log log_;
};