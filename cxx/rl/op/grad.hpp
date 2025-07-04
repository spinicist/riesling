#pragma once

#include "top.hpp"

namespace rl::TOps {

template<int ND>
struct Grad final : TOp<Cx, ND, ND + 1>
{
  TOP_INHERIT(Cx, ND, ND + 1)
  Grad(InDims const ishape, std::vector<Index> const &gradDims);
  static auto Make(InDims const ishape, std::vector<Index> const &gradDims) -> std::shared_ptr<Grad>;
  TOP_DECLARE(Grad)
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

private:
  std::vector<Index> dims_;
};

template<int ND>
struct GradVec final : TOp<Cx, ND, ND>
{
  TOP_INHERIT(Cx, ND, ND)
  GradVec(InDims const ishape, std::vector<Index> const &gradDims);
  static auto Make(InDims const ishape, std::vector<Index> const &gradDims) -> std::shared_ptr<GradVec>;
  TOP_DECLARE(GradVec)

  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

private:
  std::vector<Index> dims_;
};

} // namespace rl::TOps
