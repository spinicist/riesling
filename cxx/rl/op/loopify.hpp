#pragma once

#include "compose.hpp"
#include "loop.hpp"
#include "reshape.hpp"

namespace rl {

template <int ND> auto Loopify(typename TOps::TOp<ND + 2, 3>::Ptr op, Index const nS, Index const nTime)
  -> TOps::TOp<6, 5>::Ptr
{
  if constexpr (ND == 2) {
    auto sliceLoop = TOps::MakeLoop<2, 3>(op, nS);
    auto timeLoop = TOps::MakeLoop<5, 4>(sliceLoop, nTime);
    return timeLoop;
  } else {
    if (nS == 1) {
      auto reshape = TOps::MakeReshapeOutput(op, AddBack(op->oshape, 1));
      auto timeLoop = TOps::MakeLoop<5, 4>(reshape, nTime);
      return timeLoop;
    } else {
      throw(Log::Failure("Loopify", "Not currently supported slabs {} time {}", nS, nTime));
    }
  }
}

} // namespace rl
