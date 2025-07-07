#pragma once

#include "top.hpp"
#include "pad.hpp"

namespace rl::TOps {

template<int ND>
struct Laplacian final : TOp<ND, ND>
{
  TOP_INHERIT(ND, ND)
  Laplacian(InDims const ishape);
  static auto Make(InDims const ishape) -> std::shared_ptr<Laplacian>;
  TOP_DECLARE(Laplacian)
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;

private:
};

} // namespace rl::TOps
