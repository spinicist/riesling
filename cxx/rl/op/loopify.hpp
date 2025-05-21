#pragma once

#include "compose.hpp"
#include "loop.hpp"
#include "multiplex.hpp"
#include "reshape.hpp"

namespace rl {

template <int ND, typename Op> auto Loopify(typename Op::Ptr op, Index const nS, Index const nTime)
  -> TOps::TOp<typename Op::Scalar, Op::InRank + 1, Op::OutRank + 2>::Ptr
{
  if constexpr (ND == 2) {
    auto sliceLoop = TOps::MakeLoop(op, nS);
    auto timeLoop = TOps::MakeLoop(sliceLoop, nTime);
    return timeLoop;
  } else if constexpr (ND == 3) {
  if (nS == 1) {
    auto reshape = TOps::MakeReshapeOutput(op, AddBack(op->oshape, 1));
    auto timeLoop = TOps::MakeLoop<4, 4>(reshape, nTime);
    return timeLoop;
  } else {
    auto loop = TOps::MakeLoop<3, 3>(op, nS);
    auto slabToVol = std::make_shared<TOps::Multiplex<Cx, 5>>(op->ishape, nS);
    auto compose1 = TOps::MakeCompose(slabToVol, loop);
    auto timeLoop = TOps::MakeLoop<4, 4>(compose1, nTime);
    return timeLoop;
  }
} else {
  static_assert("Not implemented");
}
}

} // namespace rl
