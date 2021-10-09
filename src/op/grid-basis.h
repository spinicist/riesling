#pragma once

#include "../trajectory.h"
#include "operator.h"
#include <memory>

struct GridBasisOp : Operator<5, 3>
{
  GridBasisOp(Mapping map, bool const unsafe, R2 basis, Log &log);
  virtual ~GridBasisOp(){};

  virtual void A(Input const &x, Output &y) const = 0;
  virtual void Adj(Output const &x, Input &y) const = 0;

  Input::Dimensions inSize() const;
  Output::Dimensions outputDimensions() const;

  Sz3 gridDims() const;                        // Returns the dimensions of the grid
  long dimension(long const D) const;          // Returns a specific grid dimension
  Cx4 newMultichannel(long const nChan) const; // Returns a correctly sized multi-channel grid
  void setSDCExponent(float const dce);
  void setSDC(float const dc);
  void setSDC(R2 const &sdc);
  void setUnsafe();
  void setSafe();
  void sqrtOn(); // Use square-root of gridding kernel for Pipe SDC
  void sqrtOff();
  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid
  Mapping const &mapping() const;
  R2 const &basis() const;

protected:
  Mapping mapping_;
  bool safe_, sqrt_;
  Log &log_;
  float DCexp_;
  R2 basis_;
};

std::unique_ptr<GridBasisOp> make_grid_basis(
    Trajectory const &traj,
    float const os,
    bool const kb,
    bool const fastgrid,
    R2 &basis,
    Log &log,
    float const res = -1.f,
    bool const shrink = false);
