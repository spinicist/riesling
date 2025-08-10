#pragma once

#include "prox.hpp"
#include "rl/op/op.hpp"

namespace rl::Proxs {

struct L1 final : Prox
{
  PROX_INHERIT
  float       λ;
  static auto Make(float const λ, Index const sz) -> Prox::Ptr;
  static auto Make(float const λ, CMap b, Ops::Op::Ptr P) -> Prox::Ptr;
  L1(float const λ, Index const sz);
  L1(float const λ, CMap b, Ops::Op::Ptr P);
  void apply(float const α, CMap x, Map z) const;
  void conj(float const α, CMap x, Map z) const;

private:
  CMap const   b;
  Ops::Op::Ptr P;
};

struct L1I final : Prox
{
  PROX_INHERIT
  float       λ;
  static auto Make(float const λ, Index const sz) -> Prox::Ptr;
  L1I(float const λ, Index const sz);
  void apply(float const α, CMap x, Map z) const;
  void conj(float const α, CMap x, Map z) const;

private:
  Ops::Op::Ptr P;
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
  static auto Make(CMap b, Ops::Op::Ptr precon = nullptr) -> Prox::Ptr;
  SumOfSquares(CMap b, Ops::Op::Ptr precon);
  void apply(float const α, CMap x, Map z) const;
  void conj(float const α, CMap x, Map z) const;

private:
  CMap const   b;
  Ops::Op::Ptr P;
};

} // namespace rl::Proxs
