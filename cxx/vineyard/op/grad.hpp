#pragma once

#include "top.hpp"

namespace rl::TOps {

struct Grad final : TOp<Cx, 4, 5>
{
  OP_INHERIT(Cx, 4, 5)
  Grad(InDims const ishape, std::vector<Index> const &gradDims);
  OP_DECLARE(Grad)

private:
  std::vector<Index> dims_;
};

struct GradVec final : TOp<Cx, 5, 5>
{
  OP_INHERIT(Cx, 5, 5)
  GradVec(InDims const dims);
  OP_DECLARE(GradVec)
};

} // namespace rl::TOps
