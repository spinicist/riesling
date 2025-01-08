#pragma once

#include "top.hpp"

namespace rl::TOps {

template<int ND>
struct Grad final : TOp<Cx, ND, ND + 1>
{
  TOP_INHERIT(Cx, ND, ND + 1)
  Grad(InDims const ishape, std::vector<Index> const &gradDims, int const order);
  static auto Make(InDims const ishape, std::vector<Index> const &gradDims, int const order) -> std::shared_ptr<Grad>;
  TOP_DECLARE(Grad)
  void iforward(InCMap const &x, OutMap &y) const;
  void iadjoint(OutCMap const &y, InMap &x) const;

private:
  std::vector<Index> dims_;
  int mode_;
};

template<int ND>
struct GradVec final : TOp<Cx, ND, ND>
{
  TOP_INHERIT(Cx, ND, ND)
  GradVec(InDims const ishape, std::vector<Index> const &gradDims, int const order);
  static auto Make(InDims const ishape, std::vector<Index> const &gradDims, int const order) -> std::shared_ptr<GradVec>;
  TOP_DECLARE(GradVec)

private:
  std::vector<Index> dims_;
  int mode_;
};

} // namespace rl::TOps
