#pragma once
#include "log.hpp"
#include "tensorOps.hpp"
#include "tensorop.hpp"

namespace rl {

/* Scale input element-wise by factors, broadcasting as necessary.
 *
 * Equivalent to blockwise multiplication by a diagonal matrix
 * */
template <typename Scalar_, int Rank, int FrontRank = 1, int BackRank = 0>
struct TensorScale final : TensorOperator<Scalar_, Rank, Rank>
{
  OP_INHERIT(Scalar_, Rank, Rank)
  using TScales = Eigen::Tensor<Scalar, Rank - FrontRank - BackRank>;
  using Parent::adjoint;
  using Parent::forward;

  TensorScale(InDims const shape, TScales const &s, bool const inv = false)
    : Parent("Scale", shape, shape)
    , scales{s} , adjointIsInverse{inv}
  {
    for (auto ii = 0; ii < FrontRank; ii++) {
      res[0] = 1;
      brd[0] = shape[ii];
    }
    for (auto ii = FrontRank; ii < Rank - BackRank; ii++) {
      if (shape[ii] != s.dimension(ii - FrontRank)) {
        Log::Fail("Scales had shape {} expected {}", s.dimensions(), MidN<FrontRank, Rank - BackRank - FrontRank>(shape));
      }
      res[ii] = shape[ii];
      brd[ii] = 1;
    }
    for (auto ii = Rank - BackRank; ii < Rank; ii++) {
      res[ii] = 1;
      brd[ii] = shape[ii];
    }
    Log::Print<Log::Level::Debug>("{} shape {}->{} res {} brd {}", this->name, s.dimensions(), shape, res, brd);
  }

  void forward(InCMap const &x, OutMap &y) const
  {
    auto const time = this->startForward(x);
    y.device(Threads::GlobalDevice()) = x * scales.reshape(res).broadcast(brd);
    this->finishForward(y, time);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    if (adjointIsInverse) {
      inverse(y, x);
    } else {
      auto const time = this->startAdjoint(y);
      x.device(Threads::GlobalDevice()) = y * scales.reshape(res).broadcast(brd);
      this->finishAdjoint(x, time);
    }
  }

  void inverse(OutCMap const &y, InMap &x) const
  {
    auto const time = this->startInverse(y);
    x.device(Threads::GlobalDevice()) = y / scales.reshape(res).broadcast(brd);
    this->finishInverse(x, time);
  }

private:
  TScales scales;
  bool adjointIsInverse;
  Sz<Rank> res, brd;
};

} // namespace rl
