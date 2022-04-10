#pragma once

#include "precond.hpp"

struct GridBase; // Forward declare

struct SingleChannel final : Precond<Cx3>
{
  SingleChannel(GridBase *gridder);
  Cx3 apply(Cx3 const &in) const;
  Cx3 inv(Cx3 const &in) const;

private:
  R3 pre_;
};
