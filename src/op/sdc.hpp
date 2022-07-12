#pragma once

#include "../threads.h"
#include "operator.hpp"

namespace rl {

struct SDCOp final : Operator<3, 3>
{
  SDCOp(R2 const &dc, Index const nc)
    : dims_{AddFront(dc.dimensions(), nc)}
    , dc_{dc}
  {
  }

  SDCOp(Sz2 const &dims, Index const nc)
    : dims_{AddFront(dims, nc)}
  {
  }

  InputDims inputDimensions() const
  {
    return dims_;
  }

  OutputDims outputDimensions() const
  {
    return dims_;
  }

  Cx3 Adj(Cx3 const &in) const
  {
    if (dc_.size()) {
      auto const start = Log::Now();
      auto const dims = in.dimensions();
      Cx3 p(dims);
      p.device(Threads::GlobalDevice()) =
        in * dc_.cast<Cx>().reshape(Sz3{1, dc_.dimension(0), dc_.dimension(1)}).broadcast(Sz3{dims[0], 1, 1});
      Log::Debug(FMT_STRING("SDC Adjoint Took {}"), Log::ToNow(start));
      return p;
    } else {
      Log::Debug(FMT_STRING("No SDC"));
      return in;
    }
  }

private:
  Sz3 dims_;
  R2 dc_;
};
} // namespace rl
