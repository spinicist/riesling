#pragma once

#include "precond.hpp"

struct SDCPrecond
{
  R2 sdc_;

  Cx3 const apply(Cx3 const &in) const
  {
    Index const nC = in.dimension(0);
    Log::Debug(FMT_STRING("Applying SDC to {} channels"), nC);
    return in * sdc_.cast<Cx>()
                  .reshape(Sz3{1, sdc_.dimension(0), sdc_.dimension(1)})
                  .broadcast(Sz3{nC, 1, 1});
  }
};
