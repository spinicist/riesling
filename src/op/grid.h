#pragma once

#include "../kernel.h"
#include "../trajectory.h"
#include "gridBase.h"
#include "operator.h"
#include <memory>

struct GridOp : GridBase
{
  GridOp(Mapping map, bool const unsafe, Log &log);
  virtual ~GridOp(){};

  virtual void A(Input const &x, Output &y) const = 0;
  virtual void Adj(Output const &x, Input &y) const = 0;

  Input::Dimensions inputDimensions(Index const nc) const;
  Input::Dimensions inputDimensions(Index const nc, Index const ne) const;
  Output::Dimensions outputDimensions() const;

  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid

protected:
};

std::unique_ptr<GridOp> make_grid(
  Trajectory const &traj,
  float const os,
  Kernels const k,
  bool const fastgrid,
  Log &log,
  float const res = -1.f,
  bool const shrink = false);

std::unique_ptr<GridOp>
make_grid(Mapping const &mapping, Kernels const k, bool const fastgrid, Log &log);
