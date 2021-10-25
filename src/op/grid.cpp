#include "grid-kb.h"
#include "grid-nn.h"
#include "grid.h"

#include "../tensorOps.h"
#include "../threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

GridOp::GridOp(Mapping map, bool const unsafe, Log &log)
    : GridBase(map, unsafe, log)
{
}

Sz3 GridOp::outputDimensions() const
{
  return mapping_.noncartDims;
}

Cx4 GridOp::newMultichannel(long const nc) const
{
  Cx4 g(nc, mapping_.cartDims[0], mapping_.cartDims[1], mapping_.cartDims[2]);
  g.setZero();
  return g;
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

std::unique_ptr<GridOp> make_grid(
    Mapping const &mapping,
    bool const kb,
    bool const fastgrid,
    Log &log)
{
  if (kb) {
    if (mapping.type == Info::Type::ThreeD) {
      return std::make_unique<GridKB3D>(mapping, fastgrid, log);
    } else {
      return std::make_unique<GridKB2D>(mapping, fastgrid, log);
    }
  } else {
    return std::make_unique<GridNN>(mapping, fastgrid, log);
  }
}
