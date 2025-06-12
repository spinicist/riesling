#pragma once

#include "prox.hpp"

namespace rl::Proxs {

struct L1 final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  float λ;

  L1(float const λ, Index const sz);
  void apply(float const α, CMap x, Map z) const;
};

template<int O, int D>
struct L2 final : Prox<Cx>
{
  PROX_INHERIT(Cx)
  float λ;
  Index blockSize;

  L2(float const λ, Sz<O> const &shape, Sz<D> const &dims);
  void apply(float const α, CMap x, Map z) const;

private:
  Sz<O> shape;
  Sz<D> normDims;
  Sz<O - D> otherDims;
};

} // namespace rl::Proxs
