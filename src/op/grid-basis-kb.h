#pragma once

#include "../kernel.h"
#include "../trajectory.h"
#include "grid-basis.h"

template <int InPlane, int ThroughPlane>
struct GridBasisKB final : GridBasisOp
{
  GridBasisKB(
    Trajectory const &traj,
    float const os,
    bool const unsafe,
    R2 const &basis,
    Log &log,
    float const inRes = -1.f,
    bool const shrink = false);
  GridBasisKB(Mapping const &mapping, bool const unsafe, R2 const &basis, Log &log);

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

using GridBasisKB3D = GridBasisKB<3, 3>;
using GridBasisKB2D = GridBasisKB<3, 1>;
