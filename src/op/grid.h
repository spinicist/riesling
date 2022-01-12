#pragma once

#include "../kernel.h"
#include "../trajectory.h"
#include "gridBase.hpp"

#include <memory>

std::unique_ptr<GridBase> make_grid(
  Trajectory const &traj,
  float const os,
  Kernels const k,
  bool const fastgrid,
  Log &log,
  float const res = -1.f,
  bool const shrink = false);

std::unique_ptr<GridBase>
make_grid(Mapping const &mapping, Kernels const k, bool const fastgrid, Log &log);

std::unique_ptr<GridBase> make_grid_basis(
  Trajectory const &traj,
  float const os,
  Kernels const k,
  bool const fastgrid,
  R2 const &basis,
  Log &log,
  float const res = -1.f,
  bool const shrink = false);

std::unique_ptr<GridBase> make_grid_basis(
  Mapping const &mapping, Kernels const k, bool const fastgrid, R2 const &basis, Log &log);