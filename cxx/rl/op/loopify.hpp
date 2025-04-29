#pragma once

#include "compose.hpp"
#include "loop.hpp"
#include "multiplex.hpp"
#include "reshape.hpp"

namespace rl {

template <typename Op> auto Loopify(typename Op::Ptr op, Index const nSlab, Index const nTime)
  -> TOps::TOp<typename Op::Scalar, Op::InRank + 1, Op::OutRank + 2>::Ptr
{
  if (nSlab == 1) {
    auto reshape = TOps::MakeReshapeOutput(op, AddBack(op->oshape, 1));
    auto timeLoop = TOps::MakeLoop(reshape, nTime);
    return timeLoop;
  } else {
    auto loop = TOps::MakeLoop(op, nSlab);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(op->ishape, nSlab);
    auto compose1 = TOps::MakeCompose(slabToVol, loop);
    auto timeLoop = TOps::MakeLoop(compose1, nTime);
    return timeLoop;
  }
}

} // namespace rl
