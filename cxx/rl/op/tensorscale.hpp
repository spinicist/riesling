#pragma once
#include "../log/log.hpp"
#include "../tensors.hpp"
#include "top.hpp"

namespace rl::TOps {

/* Scale input element-wise by factors, broadcasting as necessary.
 *
 * Equivalent to blockwise multiplication by a diagonal matrix
 * */
template <typename Scalar_, int Rank, int FrontRank = 1, int BackRank = 0> struct TensorScale final : TOp<Scalar_, Rank, Rank>
{
  TOP_INHERIT(Scalar_, Rank, Rank)
  using TScales = Eigen::Tensor<Scalar, Rank - FrontRank - BackRank>;
  using Parent::adjoint;
  using Parent::forward;
  using Ptr = std::shared_ptr<TensorScale>;

  TensorScale(InDims const shape, TScales const &s)
    : Parent("TensorScale", shape, shape)
    , scales{s}
  {
    for (auto ii = 0; ii < FrontRank; ii++) {
      res[ii] = 1;
      brd[ii] = shape[ii];
    }
    for (auto ii = FrontRank; ii < Rank - BackRank; ii++) {
      if (shape[ii] != s.dimension(ii - FrontRank)) {
        throw Log::Failure("TOp", "Scales had shape {} expected {}", s.dimensions(),
                           MidN<FrontRank, Rank - BackRank - FrontRank>(shape));
      }
      res[ii] = shape[ii];
      brd[ii] = 1;
    }
    for (auto ii = Rank - BackRank; ii < Rank; ii++) {
      res[ii] = 1;
      brd[ii] = shape[ii];
    }
    Log::Debug("TOp", "TensorScale weights {} reshape {} broadcast {}", scales.dimensions(), res, brd);
  }

  void forward(InCMap const x, OutMap y) const
  {
    auto const time = this->startForward(x, y, false);
    y.device(Threads::TensorDevice()) = x * scales.reshape(res).broadcast(brd);
    this->finishForward(y, time, false);
  }

  void iforward(InCMap const x, OutMap y) const
  {
    auto const time = this->startForward(x, y, true);
    y.device(Threads::TensorDevice()) += x * scales.reshape(res).broadcast(brd);
    this->finishForward(y, time, false);
  }

  void adjoint(OutCMap const y, InMap x) const
  {
    auto const time = this->startAdjoint(y, x, false);
    x.device(Threads::TensorDevice()) = y * scales.reshape(res).broadcast(brd);
    this->finishAdjoint(x, time, false);
  }

  void iadjoint(OutCMap const y, InMap x) const
  {
    auto const time = this->startAdjoint(y, x, true);
    x.device(Threads::TensorDevice()) += y * scales.reshape(res).broadcast(brd);
    this->finishAdjoint(x, time, false);
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

  auto weights() const -> TScales const & { return scales; }

private:
  TScales  scales;
  Sz<Rank> res, brd;
};

} // namespace rl::TOps
