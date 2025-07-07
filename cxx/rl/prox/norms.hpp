#pragma once

#include "prox.hpp"

namespace rl::Proxs {

struct L1 final : Prox
{
  PROX_INHERIT
  float       λ;
  static auto Make(float const λ, Index const sz) -> Prox::Ptr;
  L1(float const λ, Index const sz);
  void primal(float const α, CMap x, Map z) const;
  void dual(float const α, CMap x, Map z) const;
};

template <int O, int D> struct L2 final : Prox
{
  PROX_INHERIT
  float       λ;
  Index       blockSize;
  static auto Make(float const λ, Sz<O> const &shape, Sz<D> const &dims) -> Prox::Ptr;
  L2(float const λ, Sz<O> const &shape, Sz<D> const &dims);
  void primal(float const α, CMap x, Map z) const;
  void dual(float const α, CMap x, Map z) const;

private:
  Sz<O>     shape;
  Sz<D>     normDims;
  Sz<O - D> otherDims;
};

} // namespace rl::Proxs
