#pragma once

#include "../types.hpp"
#include "../kernel/expsemi.hpp"
#include "../kernel/tophat.hpp"
#include "top.hpp"

namespace rl::TOps {

template <int ND, int ED, typename KF> struct Apodize final : TOp<Cx, ND + ED, ND + ED>
{
  TOP_INHERIT(Cx, ND + ED, ND + ED)
  Apodize(Sz<ND + ED> const shape, Sz<ND + ED> const gshape, float const osamp);
  TOP_DECLARE(Apodize)

  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;
};

template <int ND, int ED> struct Apodize<ND, ED, ExpSemi<4>> final : TOp<Cx, ND + ED, ND + ED>
{
  TOP_INHERIT(Cx, ND + ED, ND + ED)
  Apodize(Sz<ND + ED> const shape, Sz<ND + ED> const gshape, float const osamp);
  TOP_DECLARE(Apodize)

  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;

private:
  InTensor apo_;
  InDims   apoBrd_, padLeft_;
  std::array<std::pair<Index, Index>, ND + ED> paddings_;
};

template <int ND, int ED> struct Apodize<ND, ED, TopHat<1>> final : TOp<Cx, ND + ED, ND + ED>
{
  TOP_INHERIT(Cx, ND + ED, ND + ED)
  Apodize(Sz<ND + ED> const shape, Sz<ND + ED> const gshape, float const osamp);
  TOP_DECLARE(Apodize)

  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const;
  void iforward(InCMap x, OutMap y, float const s = 1.f) const;

private:
  InDims   apoBrd_, padLeft_;
  std::array<std::pair<Index, Index>, ND + ED> paddings_;
};


} // namespace rl
