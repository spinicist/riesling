#pragma once

#include "../op/grid.hpp"
#include "../threads.h"
#include "precond.hpp"

struct SingleChannel final : Precond
{
  SingleChannel(GridBase *gridder)
    : Precond{}
  {
    Cx3 W(AddFront(LastN<2>(gridder->outputDimensions()), 1));
    W.setConstant(1.f);
    W = gridder->A(gridder->Adj(W));
    pre_ = (W.real() > 0.f).select(W.real().inverse(), W.constant(0.f).real());
    Log::Print("Created single-channel pre-conditioner");
    Log::Image(pre_, "pre");
  }

  Cx3 const apply(Cx3 const &in) const
  {
    Log::Debug("Applying single-channel pre-conditioner");
    Index const nC = in.dimension(0);
    Cx3 p(in.dimensions());
    p.device(Threads::GlobalDevice()) = in * pre_.cast<Cx>().broadcast(Sz3{nC, 1, 1});
    return p;
  }

private:
  R3 pre_;
};
