#pragma once

#include "kernel/kernel.hpp"
#include "mapping.hpp"
#include "tensorop.hpp"
#include "threads.hpp"

#include <mutex>

namespace rl {

auto IdBasis() -> Re2;

template <typename Scalar_, size_t NDim>
struct Grid final : TensorOperator<Scalar_, NDim + 2, 3>
{
  OP_INHERIT(Scalar_, NDim + 2, 3)
  using Parent::adjoint;
  using Parent::forward;

  std::shared_ptr<Kernel<Scalar, NDim>> kernel;
  Mapping<NDim>                         mapping;
  Re2                                   basis;

  static auto Make(Trajectory const &t,
                   std::string const kt,
                   float const       os,
                   Index const       nC = 1,
                   Re2 const        &b = IdBasis(),
                   Index const       bSz = 32,
                   Index const       sSz = 16384) -> std::shared_ptr<Grid<Scalar, NDim>>;
  Grid(std::shared_ptr<Kernel<Scalar, NDim>> const &k, Mapping<NDim> const m, Index const nC, Re2 const &b = IdBasis());
  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;
};

} // namespace rl
