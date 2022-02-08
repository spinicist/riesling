#pragma once

#include "precond.hpp"

struct SDCPrecond
{
  R2 sdc_;
  Index channels_;

  Sz3 dimensions() const
  {
    return AddFront(sdc_.dimensions(), channels_);
  }

  Cx3 const apply(Cx3 const &in) const
  {
    return apply(in, channels_);
  }

  Cx3 const apply(Cx3 const &in, Index const ncIn) const
  {
    Index const nc = (ncIn > 0) ? ncIn : channels_;
    return in * sdc_.cast<Cx>()
                  .reshape(Sz3{1, sdc_.dimension(0), sdc_.dimension(1)})
                  .broadcast(Sz3{nc, 1, 1});
  }
};
