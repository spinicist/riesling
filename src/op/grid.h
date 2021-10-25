#pragma once

#include "../trajectory.h"
#include "gridBase.h"
#include "operator.h"
#include <memory>

struct GridOp : Operator<4, 3>, GridBase
{
  GridOp(Mapping map, bool const unsafe, Log &log);
  virtual ~GridOp(){};

  virtual void A(Input const &x, Output &y) const = 0;
  virtual void Adj(Output const &x, Input &y) const = 0;

  Input::Dimensions inSize() const;
  Output::Dimensions outputDimensions() const;

  Cx4 newMultichannel(long const nChan) const; // Returns a correctly sized multi-channel grid
  virtual R3 apodization(Sz3 const sz) const = 0; // Calculate the apodization factor for this grid

protected:
};

std::unique_ptr<GridOp> make_grid(
    Trajectory const &traj,
    float const os,
    bool const kb,
    bool const fastgrid,
    Log &log,
    float const res = -1.f,
    bool const shrink = false);

std::unique_ptr<GridOp> make_grid(
    Mapping const &mapping,
    bool const kb,
    bool const fastgrid,
    Log &log);
