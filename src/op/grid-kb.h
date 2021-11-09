#pragma once

#include "../kernel.h"
#include "../trajectory.h"
#include "grid.h"

template <int InPlane, int ThroughPlane>
struct GridKB final : GridOp
{
  GridKB(
    Trajectory const &traj,
    float const os,
    bool const unsafe,
    Log &log,
    float const inRes = -1.f,
    bool const shrink = false);
  GridKB(Mapping const &mapping, bool const unsafe, Log &log);
  void A(Input const &x, Output &y) const;
  void Adj(Output const &x, Input &y) const;

  R3 apodization(Sz3 const sz) const;

  void sqrtOn();
  void sqrtOff();

private:
  using FixIn = Eigen::type2index<InPlane>;
  using FixThrough = Eigen::type2index<ThroughPlane>;

  Kernel<InPlane, ThroughPlane> kernel_;
};

using GridKB3D = GridKB<3, 3>;
using GridKB2D = GridKB<3, 1>;
