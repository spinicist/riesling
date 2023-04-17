#pragma once

#include "log.hpp"
#include "op/operator.hpp"
#include <optional>

namespace rl {

struct PowerReturn {
  float val;
  Eigen::VectorXcf vec;
};

auto PowerMethodForward(std::shared_ptr<Op::Operator<Cx>> op, std::shared_ptr<Op::Operator<Cx>> M, Index const iterLimit) -> PowerReturn;
auto PowerMethodAdjoint(std::shared_ptr<Op::Operator<Cx>> op, std::shared_ptr<Op::Operator<Cx>> M, Index const iterLimit) -> PowerReturn;

} // namespace rl
