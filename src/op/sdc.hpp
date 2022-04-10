#pragma once

#include "../threads.h"
#include "operator.hpp"

struct SDCOp final : Operator<3, 3>
{
  SDCOp(R2 const &dc, Index const nc)
    : dc_{dc}
    , nc_{nc}
  {
  }

  InputDims inputDimensions() const
  {
    return AddFront(dc_.dimensions(), nc_);
  }

  OutputDims outputDimensions() const
  {
    return AddFront(dc_.dimensions(), nc_);
  }

  template <typename T>
  auto Adj(T const &in) const
  {
    Log::Debug(FMT_STRING("Applying SDC to {} channels"), nc_);
    Cx3 p(outputDimensions());
    p.device(Threads::GlobalDevice()) =
      in *
      dc_.cast<Cx>().reshape(Sz3{1, dc_.dimension(0), dc_.dimension(1)}).broadcast(Sz3{nc_, 1, 1});
    return p;
  }

private:
  R2 dc_;
  Index nc_;
};
