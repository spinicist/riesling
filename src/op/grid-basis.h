#pragma once

#include "../kernel.h"
#include "../trajectory.h"
#include "gridBase.h"
#include "operator.h"
#include <memory>

struct GridBasisOp : GridBase
{
  GridBasisOp(Mapping map, bool const unsafe, R2 basis, Log &log);
  virtual ~GridBasisOp(){};

  virtual void A(Input const &x, Output &y) const = 0;
  virtual void Adj(Output const &x, Input &y) const = 0;

  Input::Dimensions inputDimensions(Index const nc) const;
  Output::Dimensions outputDimensions() const;

  Index dimension(Index const D) const; // Returns a specific grid dimension
  R2 const &basis() const;

protected:
  R2 basis_;
  float basisScale_;
};

std::unique_ptr<GridBasisOp> make_grid_basis(
  Trajectory const &traj,
  float const os,
  Kernels const k,
  bool const fastgrid,
  R2 const &basis,
  Log &log,
  float const res = -1.f,
  bool const shrink = false);

std::unique_ptr<GridBasisOp> make_grid_basis(
  Mapping const &mapping, Kernels const k, bool const fastgrid, R2 const &basis, Log &log);
