#pragma once

#include "../trajectory.h"
#include "grid.h"

struct GridNN final : GridOp
{
  GridNN(
      Trajectory const &traj,
      float const os,
      bool const unsafe,
      Log &log,
      float const inRes = -1.f,
      bool const shrink = false);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;

  R3 apodization(Sz3 const sz) const;
};
