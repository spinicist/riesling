#pragma once

#include "../threads.h"
#include "precond.hpp"

struct SDCPrecond final : Precond
{
  SDCPrecond(R2 const &dc)
    : Precond{}
    , dc_{dc}
  {
  }

  Cx3 const apply(Cx3 const &in) const
  {
    Index const nC = in.dimension(0);
    Log::Debug(FMT_STRING("Applying SDC to {} channels"), nC);
    Cx3 p(in.dimensions());
    p.device(Threads::GlobalDevice()) =
      in *
      dc_.cast<Cx>().reshape(Sz3{1, dc_.dimension(0), dc_.dimension(1)}).broadcast(Sz3{nC, 1, 1});
    return p;
  }

private:
  R2 dc_;
};
