#pragma once

#include "tensorop.hpp"

#include "fft/fft.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

namespace rl::Ops {

template <int Rank, int FFTRank>
struct FFTOp final : TensorOperator<Cx, Rank, Rank>
{
  OP_INHERIT(Cx, Rank, Rank)

  FFTOp(InDims const &dims);
  FFTOp(InMap x);

  using Parent::adjoint;
  using Parent::forward;

  void forward(InCMap const &x, OutMap &y) const;
  void adjoint(OutCMap const &y, InMap &x) const;

private:
  InDims                                   dims_;
  std::shared_ptr<FFT::FFT<Rank, FFTRank>> fft_;
};
} // namespace rl::Ops
