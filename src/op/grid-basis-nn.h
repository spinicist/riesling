#pragma once

#include "../trajectory.h"
#include "grid-basis.h"

struct GridBasisNN final : GridBasisOp
{
  GridBasisNN(
      Trajectory const &traj,
      float const os,
      bool const unsafe,
      R2 &basis,
      Log &log,
      float const inRes = -1.f,
      bool const shrink = false);
  GridBasisNN(
      Mapping const &map,
      bool const unsafe,
      R2 &basis,
      Log &log);

  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;

  R3 apodization(Sz3 const sz) const;

};
