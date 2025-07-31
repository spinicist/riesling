#pragma once

#include "prox.hpp"

namespace rl::Proxs {

struct L1 final : Prox
{
  PROX_INHERIT
  float       λ;
  static auto Make(float const λ, Index const sz) -> Prox::Ptr;
  L1(float const λ, Index const sz);
  void apply(float const α, CMap x, Map z) const;
  void conj(float const α, CMap x, Map z) const;
};

template <int O, int D> struct L2 final : Prox
{
  PROX_INHERIT
  float       λ;
  Index       blockSize;
  static auto Make(float const λ, Sz<O> const &shape, Sz<D> const &dims) -> Prox::Ptr;
  L2(float const λ, Sz<O> const &shape, Sz<D> const &dims);
  void apply(float const α, CMap x, Map z) const;
  void conj(float const α, CMap x, Map z) const;

private:
  Sz<O>     shape;
  Sz<D>     normDims;
  Sz<O - D> otherDims;
};

struct SumOfSquares final : Prox
{
  PROX_INHERIT
  static auto Make(CMap b) -> Prox::Ptr;
  SumOfSquares(CMap b);
  void apply(float const α, CMap x, Map z) const;
  void conj(float const α, CMap x, Map z) const;

private:
  CMap const b;
};

} // namespace rl::Proxs
