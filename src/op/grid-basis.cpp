#include "grid-basis-kernel.hpp"
#include "grid-basis-nn.h"
#include "grid-basis.h"

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

Index GridBasisOp::dimension(Index const D) const
{
  assert(D < 3);
  return mapping_.cartDims[D];
}

Sz5 GridBasisOp::inputDimensions(Index const nc) const
{
  return Sz5{
    nc, basis_.dimension(1), mapping_.cartDims[0], mapping_.cartDims[1], mapping_.cartDims[2]};
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
  Kernels const k,
  bool const fastgrid,
  R2 const &basis,
  Log &log,
  float const res,
  bool const shrink)
{
  switch (k) {
  case Kernels::NN:
    return std::make_unique<GridBasisNN>(traj, os, fastgrid, basis, log, res, shrink);
  case Kernels::KB3:
    if (traj.info().type == Info::Type::ThreeD) {
      return std::make_unique<GridBasis<KaiserBessel<3, 3>>>(
        traj, os, fastgrid, basis, log, res, shrink);
    } else {
      return std::make_unique<GridBasis<KaiserBessel<3, 1>>>(
        traj, os, fastgrid, basis, log, res, shrink);
    }
  case Kernels::KB5:
    if (traj.info().type == Info::Type::ThreeD) {
      return std::make_unique<GridBasis<KaiserBessel<5, 5>>>(
        traj, os, fastgrid, basis, log, res, shrink);
    } else {
      return std::make_unique<GridBasis<KaiserBessel<5, 1>>>(
        traj, os, fastgrid, basis, log, res, shrink);
    }
  }
  __builtin_unreachable();
}

std::unique_ptr<GridBasisOp> make_grid_basis(
  Mapping const &mapping, Kernels const k, bool const fastgrid, R2 const &basis, Log &log)
{
  switch (k) {
  case Kernels::NN:
    return std::make_unique<GridBasisNN>(mapping, fastgrid, basis, log);
  case Kernels::KB3:
    if (mapping.type == Info::Type::ThreeD) {
      return std::make_unique<GridBasis<KaiserBessel<3, 3>>>(mapping, fastgrid, basis, log);
    } else {
      return std::make_unique<GridBasis<KaiserBessel<3, 1>>>(mapping, fastgrid, basis, log);
    }
  case Kernels::KB5:
    if (mapping.type == Info::Type::ThreeD) {
      return std::make_unique<GridBasis<KaiserBessel<5, 5>>>(mapping, fastgrid, basis, log);
    } else {
      return std::make_unique<GridBasis<KaiserBessel<5, 1>>>(mapping, fastgrid, basis, log);
    }
  }
  __builtin_unreachable();
}
