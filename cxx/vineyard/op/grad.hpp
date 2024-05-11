#pragma once

#include "top.hpp"

namespace rl {

struct GradOp final : TOp<Cx, 4, 5>
{
  OP_INHERIT(Cx, 4, 5)
  GradOp(InDims const ishape, std::vector<Index> const &gradDims);
  OP_DECLARE(GradOp)

private:
  std::vector<Index> dims_;
};

struct GradVecOp final : TOp<Cx, 5, 5>
{
  OP_INHERIT(Cx, 5, 5)
  GradVecOp(InDims const dims);
  OP_DECLARE(GradVecOp)
};

} // namespace rl
