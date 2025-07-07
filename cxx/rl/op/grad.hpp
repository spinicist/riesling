#pragma once

#include "top.hpp"

namespace rl::TOps {

template <int ND, int NG> struct Grad final : TOp<Cx, ND, ND + 1>
{
  TOP_INHERIT(Cx, ND, ND + 1)
  Grad(InDims const ishape, Sz<NG> constgradDims, int const order);
  static auto Make(InDims const ishape, Sz<NG> const gradDims, int const order) -> std::shared_ptr<Grad>;
  TOP_DECLARE(Grad)
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

private:
  Sz<NG> const dims_;
  int const    mode_;
};

template <int ND, int NG> struct Div final : TOp<Cx, ND + 1, ND>
{
  TOP_INHERIT(Cx, ND + 1, ND)
  Div(OutDims const ishape, Sz<NG> constgradDims, int const order);
  static auto Make(OutDims const ishape, Sz<NG> constgradDims, int const order) -> std::shared_ptr<Div>;
  TOP_DECLARE(Div)
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

private:
  Sz<NG> const dims_;
  int const    mode_;
};

template <int ND, int NG> struct GradVec final : TOp<Cx, ND, ND>
{
  TOP_INHERIT(Cx, ND, ND)
  GradVec(InDims const ishape, Sz<NG> constgradDims, int const order);
  static auto Make(InDims const ishape, Sz<NG> constgradDims, int const order) -> std::shared_ptr<GradVec>;
  TOP_DECLARE(GradVec)

  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

private:
  Sz<NG> const dims_;
  int const    mode_;
};

} // namespace rl::TOps
