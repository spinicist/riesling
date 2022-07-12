#pragma once

#include "../kernel.hpp"
#include "../trajectory.h"
#include "precond.hpp"

namespace rl {

struct SingleChannel final : Precond<Cx3>
{
  SingleChannel(Trajectory const &traj, Kernel const *k);
  Cx3 apply(Cx3 const &in) const;
  Cx3 inv(Cx3 const &in) const;

private:
  R3 pre_;
};

}
