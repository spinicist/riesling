#include "grid-basis.h"
#include "grid-basis-kb.h"
#include "grid-basis-nn.h"

#include "../tensorOps.h"
#include "../threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

GridBasisOp::GridBasisOp(Mapping map, bool const unsafe, R2 basis, Log &log)
  : GridBase(map, unsafe, log)
  , basis_{basis}
  , basisScale_{std::sqrt((float)basis_.dimension(0))}
{
  log_.info("Basis size {}x{}, scale {}", basis_.dimension(0), basis_.dimension(1), basisScale_);
}

long GridBasisOp::dimension(long const D) const
{
  assert(D < 3);
  return mapping_.cartDims[D];
}

Sz3 GridBasisOp::outputDimensions() const
{
  return mapping_.noncartDims;
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
  R2 const &basis,
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

std::unique_ptr<GridBasisOp> make_grid_basis(
  Mapping const &mapping, bool const kb, bool const fastgrid, R2 const &basis, Log &log)
{
  if (kb) {
    if (mapping.type == Info::Type::ThreeD) {
      return std::make_unique<GridBasisKB3D>(mapping, fastgrid, basis, log);
    } else {
      return std::make_unique<GridBasisKB2D>(mapping, fastgrid, basis, log);
    }
  } else {
    return std::make_unique<GridBasisNN>(mapping, fastgrid, basis, log);
  }
}
