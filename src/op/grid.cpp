#include "grid.h"
#include "grid-basis-kernel.hpp"
#include "grid-basis-nn.hpp"
#include "grid-echo-kernel.hpp"
#include "grid-echo-nn.hpp"

#include "../tensorOps.h"
#include "../threads.h"

std::unique_ptr<GridBase> make_grid(
  Trajectory const &traj,
  float const os,
  Kernels const k,
  bool const fastgrid,
  Log &log,
  float const res,
  bool const shrink)
{
  switch (k) {
  case Kernels::NN:
    return std::make_unique<GridNN>(traj, os, fastgrid, log, res, shrink);
  case Kernels::KB3:
    if (traj.info().type == Info::Type::ThreeD) {
      return std::make_unique<Grid<KaiserBessel<3, 3>>>(traj, os, fastgrid, log, res, shrink);
    } else {
      return std::make_unique<Grid<KaiserBessel<3, 1>>>(traj, os, fastgrid, log, res, shrink);
    }
  case Kernels::KB5:
    if (traj.info().type == Info::Type::ThreeD) {
      return std::make_unique<Grid<KaiserBessel<5, 5>>>(traj, os, fastgrid, log, res, shrink);
    } else {
      return std::make_unique<Grid<KaiserBessel<5, 1>>>(traj, os, fastgrid, log, res, shrink);
    }
  }
  __builtin_unreachable();
}

std::unique_ptr<GridBase>
make_grid(Mapping const &mapping, Kernels const k, bool const fastgrid, Log &log)
{
  switch (k) {
  case Kernels::NN:
    return std::make_unique<GridNN>(mapping, fastgrid, log);
  case Kernels::KB3:
    if (mapping.type == Info::Type::ThreeD) {
      return std::make_unique<Grid<KaiserBessel<3, 3>>>(mapping, fastgrid, log);
    } else {
      return std::make_unique<Grid<KaiserBessel<3, 1>>>(mapping, fastgrid, log);
    }
  case Kernels::KB5:
    if (mapping.type == Info::Type::ThreeD) {
      return std::make_unique<Grid<KaiserBessel<5, 5>>>(mapping, fastgrid, log);
    } else {
      return std::make_unique<Grid<KaiserBessel<5, 1>>>(mapping, fastgrid, log);
    }
  }
  __builtin_unreachable();
}

std::unique_ptr<GridBase> make_grid_basis(
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

std::unique_ptr<GridBase> make_grid_basis(
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
