#pragma once
#include "log.hpp"
#include "tensors.hpp"
#include "top.hpp"

namespace rl {

/* Scale input element-wise by factors, broadcasting as necessary.
 *
 * Equivalent to blockwise multiplication by a diagonal matrix
 * */
template <typename Scalar_, int Rank, int FrontRank = 1, int BackRank = 0>
struct TensorScale final : TOp<Scalar_, Rank, Rank>
{
  OP_INHERIT(Scalar_, Rank, Rank)
  using TScales = Eigen::Tensor<Scalar, Rank - FrontRank - BackRank>;
  using Parent::adjoint;
  using Parent::forward;

  TensorScale(InDims const shape, TScales const &s)
    : Parent("TensorDiag", shape, shape)
    , scales{s}
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
    Log::Debug("{} shape {}->{} res {} brd {}", this->name, s.dimensions(), shape, res, brd);
  }

  void forward(InCMap const &x, OutMap &y) const
  {
    auto const time = this->startForward(x);
    y.device(Threads::GlobalDevice()) = x * scales.reshape(res).broadcast(brd);
    this->finishForward(y, time);
  }

  void adjoint(OutCMap const &y, InMap &x) const
  {
    auto const time = this->startAdjoint(y);
    x.device(Threads::GlobalDevice()) = y * scales.reshape(res).broadcast(brd);
    this->finishAdjoint(x, time);
  }

  auto inverse() const -> std::shared_ptr<Ops::Op<Cx>>
  {
    TScales inv = 1.f / scales;
    return std::make_shared<TensorScale<Scalar_, Rank, FrontRank, BackRank>>(this->ishape, scales);
  }

  auto operator+(Scalar const s) const -> std::shared_ptr<Ops::Op<Cx>>
  {
    TScales p = scales + s;
    return std::make_shared<TensorScale<Scalar_, Rank, FrontRank, BackRank>>(this->ishape, p);
  }

private:
  TScales  scales;
  Sz<Rank> res, brd;
};

} // namespace rl
