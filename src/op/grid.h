#pragma once

#include "../trajectory.h"
#include "operator.h"
#include <memory>

struct GridOp : Operator<4, 3>
{
  GridOp(Mapping map, bool const unsafe, Log &log);
  virtual ~GridOp(){};

  virtual void A(Input const &x, Output &y) const = 0;
  virtual void Adj(Output const &x, Input &y) const = 0;

  Input::Dimensions inSize() const;
  Output::Dimensions outSize() const;

  Sz3 gridDims() const;                        // Returns the dimensions of the grid
  Cx4 newMultichannel(long const nChan) const; // Returns a correctly sized multi-channel grid
  void setSDCExponent(float const dce);
  void setSDC(float const dc);
  void setSDC(R2 const &sdc);
  void setUnsafe();
  void setSafe();
  void sqrtOn(); // Use square-root of gridding kernel for Pipe SDC
  void sqrtOff();
  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid

protected:
  Mapping mapping_;
  bool safe_, sqrt_;
  Log &log_;
  float DCexp_;
};

std::unique_ptr<GridOp> make_grid(
    Trajectory const &traj,
    float const os,
    bool const kb,
    bool const fastgrid,
    Log &log,
    float const res = -1.f,
    bool const shrink = false);
