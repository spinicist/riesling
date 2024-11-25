#pragma once

#include "top.hpp"

namespace rl::TOps {

template<int ND>
struct Grad final : TOp<Cx, ND, ND + 1>
{
  TOP_INHERIT(Cx, ND, ND + 1)
  Grad(InDims const ishape, std::vector<Index> const &gradDims);
  TOP_DECLARE(Grad)

private:
  std::vector<Index> dims_;
};

template<int ND>
struct GradVec final : TOp<Cx, ND, ND>
{
  TOP_INHERIT(Cx, ND, ND)
  GradVec(InDims const ishape, std::vector<Index> const &gradDims);
  TOP_DECLARE(GradVec)

private:
  std::vector<Index> dims_;
};

} // namespace rl::TOps
