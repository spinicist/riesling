#include "grid.h"
#include "grid-kb.h"
#include "grid-nn.h"

#include "../tensorOps.h"
#include "../threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

GridOp::GridOp(Mapping map, bool const unsafe, Log &log)
    : mapping_{std::move(map)}
    , safe_{!unsafe}
    , log_{log}
    , DCexp_{1.f}
{
}

Sz3 GridOp::gridDims() const
{
  return mapping_.cartDims;
}

Cx4 GridOp::newMultichannel(long const nc) const
{
  Cx4 g(nc, mapping_.cartDims[0], mapping_.cartDims[1], mapping_.cartDims[2]);
  g.setZero();
  return g;
}

void GridOp::setSDC(float const d)
{
  std::fill(mapping_.sdc.begin(), mapping_.sdc.end(), d);
}

void GridOp ::setSDC(R2 const &sdc)
{
  std::transform(
      mapping_.noncart.begin(),
      mapping_.noncart.end(),
      mapping_.sdc.begin(),
      [&sdc](NoncartesianIndex const &nc) { return sdc(nc.read, nc.spoke); });
}

void GridOp::setSDCExponent(float const dce)
{
  DCexp_ = dce;
}

void GridOp::setUnsafe()
{
  safe_ = true;
}

void GridOp::setSafe()
{
  safe_ = false;
}

void GridOp::sqrtOn()
{
  sqrt_ = true;
}

void GridOp::sqrtOff()
{
  sqrt_ = false;
}

std::unique_ptr<GridOp> make_grid(
    Trajectory const &traj,
    float const os,
    bool const kb,
    bool const fastgrid,
    Log &log,
    float const res,
    bool const shrink)
{
  if (kb) {
    if (traj.info().type == Info::Type::ThreeD) {
      return std::make_unique<GridKB3D>(traj, os, fastgrid, log, res, shrink);
    } else {
      return std::make_unique<GridKB2D>(traj, os, fastgrid, log, res, shrink);
    }
  } else {
    return std::make_unique<GridNN>(traj, os, fastgrid, log, res, shrink);
  }
}
