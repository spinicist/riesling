#pragma once

#include "../log/log.hpp"
#include "../op/ops.hpp"
#include <optional>

namespace rl {

struct PowerReturn
{
  float   val;
  VectorX vec;
};

auto PowerMethodForward(std::shared_ptr<Ops::Op> op, std::shared_ptr<Ops::Op> M, Index const iterLimit) -> PowerReturn;
auto PowerMethodAdjoint(std::shared_ptr<Ops::Op> op, std::shared_ptr<Ops::Op> M, Index const iterLimit) -> PowerReturn;

} // namespace rl
