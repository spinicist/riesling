#pragma once

#include "log.hpp"
#include "op/operator.hpp"
#include <optional>

namespace rl {

struct PowerReturn {
  float val;
  Eigen::VectorXcf vec;
};

auto PowerMethodForward(std::shared_ptr<LinOps::Op<Cx>> op, std::shared_ptr<LinOps::Op<Cx>> M, Index const iterLimit) -> PowerReturn;
auto PowerMethodAdjoint(std::shared_ptr<LinOps::Op<Cx>> op, std::shared_ptr<LinOps::Op<Cx>> M, Index const iterLimit) -> PowerReturn;

} // namespace rl
