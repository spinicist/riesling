#pragma once

#include "top.hpp"
#include "pad.hpp"

namespace rl::TOps {

template<int ND>
struct Grad2 final : TOp<Cx, ND, ND>
{
  TOP_INHERIT(Cx, ND, ND)
  Grad2(InDims const ishape);
  static auto Make(InDims const ishape) -> std::shared_ptr<Grad2>;
  TOP_DECLARE(Grad2)
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

private:
  Pad<Cx, ND> pad;
  void filter(OutMap x) const;
};

} // namespace rl::TOps
