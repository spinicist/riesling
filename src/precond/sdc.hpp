#pragma once

#include "../types.h"

struct SDCPrecond
{
  R2 sdc_;
  Index channels_;

  Sz3 dimensions() const
  {
    return AddFront(sdc_.dimensions(), channels_);
  }

  auto operator()(Cx3 const &in, Index const ncIn = 0) const
  {
    Index const nc = (ncIn > 0) ? ncIn : channels_;
    return in * sdc_.cast<Cx>()
                  .reshape(Sz3{1, sdc_.dimension(0), sdc_.dimension(1)})
                  .broadcast(Sz3{nc, 1, 1});
  }
};
