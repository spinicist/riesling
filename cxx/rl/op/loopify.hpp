#pragma once

#include "compose.hpp"
#include "loop.hpp"
#include "reshape.hpp"

namespace rl {

template <typename Op> auto Loopify(typename Op::Ptr op, Index const nS, Index const nTime)
  -> TOps::TOp<typename Op::Scalar, Op::InRank + 1, Op::OutRank + 2>::Ptr
{
  if (nS == 1) {
    auto reshape = TOps::MakeReshapeOutput(op, AddBack(op->oshape, 1));
    auto timeLoop = TOps::MakeLoop<4, 4>(reshape, nTime);
    return timeLoop;
  } else {
    throw(Log::Failure("Loopify", "Not currently supported"));
  }
}

} // namespace rl
