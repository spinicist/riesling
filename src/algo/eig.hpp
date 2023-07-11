#pragma once

#include "log.hpp"
#include "op/ops.hpp"
#include <optional>

namespace rl {

struct PowerReturn
{
  float            val;
  Eigen::VectorXcf vec;
};

auto PowerMethod(std::shared_ptr<Ops::Op<Cx>> A, Index const iterLimit) -> PowerReturn;
auto PowerMethodForward(std::shared_ptr<Ops::Op<Cx>> op, std::shared_ptr<Ops::Op<Cx>> M, Index const iterLimit) -> PowerReturn;
auto PowerMethodAdjoint(std::shared_ptr<Ops::Op<Cx>> op, std::shared_ptr<Ops::Op<Cx>> M, Index const iterLimit) -> PowerReturn;

} // namespace rl
