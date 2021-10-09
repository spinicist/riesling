#include "grid-basis-kb.h"
#include "grid-basis-nn.h"
#include "grid-basis.h"

#include "../tensorOps.h"
#include "../threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

GridBasisOp::GridBasisOp(Mapping map, bool const unsafe, R2 basis, Log &log)
    : mapping_{std::move(map)}
    , safe_{!unsafe}
    , sqrt_{false}
    , log_{log}
    , DCexp_{1.f}
    , basis_{basis}
{
  log_.info("Basis size {}x{}", basis_.dimension(0), basis_.dimension(1));
}

long GridBasisOp::dimension(long const D) const
{
  assert(D < 3);
  return mapping_.cartDims[D];
}

Sz3 GridBasisOp::gridDims() const
{
  return mapping_.cartDims;
}

Sz3 GridBasisOp::outputDimensions() const
{
  return mapping_.noncartDims;
}

Cx4 GridBasisOp::newMultichannel(long const nc) const
{
  Cx4 g(nc, mapping_.cartDims[0], mapping_.cartDims[1], mapping_.cartDims[2]);
  g.setZero();
  return g;
}

void GridBasisOp::setSDC(float const d)
{
  std::fill(mapping_.sdc.begin(), mapping_.sdc.end(), d);
}

void GridBasisOp ::setSDC(R2 const &sdc)
{
  std::transform(
      mapping_.noncart.begin(),
      mapping_.noncart.end(),
      mapping_.sdc.begin(),
      [&sdc](NoncartesianIndex const &nc) { return sdc(nc.read, nc.spoke); });
}

void GridBasisOp::setSDCExponent(float const dce)
{
  DCexp_ = dce;
}

void GridBasisOp::setUnsafe()
{
  safe_ = true;
}

void GridBasisOp::setSafe()
{
  safe_ = false;
}

void GridBasisOp::sqrtOn()
{
  sqrt_ = true;
}

void GridBasisOp::sqrtOff()
{
  sqrt_ = false;
}

Mapping const &GridBasisOp::mapping() const
{
  return mapping_;
}

R2 const &GridBasisOp::basis() const
{
  return basis_;
}

std::unique_ptr<GridBasisOp> make_grid_basis(
    Trajectory const &traj,
    float const os,
    bool const kb,
    bool const fastgrid,
    R2 &basis,
    Log &log,
    float const res,
    bool const shrink)
{
  if (kb) {
    if (traj.info().type == Info::Type::ThreeD) {
      return std::make_unique<GridBasisKB3D>(traj, os, fastgrid, basis, log, res, shrink);
    } else {
      return std::make_unique<GridBasisKB2D>(traj, os, fastgrid, basis, log, res, shrink);
    }
  } else {
    return std::make_unique<GridBasisNN>(traj, os, fastgrid, basis, log, res, shrink);
  }
}
