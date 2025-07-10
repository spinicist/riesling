#pragma once
#include "../log/log.hpp"
#include "../tensors.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

/* Scale input element-wise by factors, broadcasting as necessary.
 *
 * Equivalent to blockwise multiplication by a diagonal matrix
 * */
template <int Rank, int FrontRank = 1, int BackRank = 0> struct TensorScale final : TOp<Rank, Rank>
{
  TOP_INHERIT(Rank, Rank)
  using TScales = Eigen::Tensor<Cx, Rank - FrontRank - BackRank>;
  using Parent::adjoint;
  using Parent::forward;
  using Ptr = std::shared_ptr<TensorScale>;

  static auto Make(InDims const shape, TScales const &s) -> Ptr { return std::make_shared<TensorScale>(shape, s); }

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

  void forward(InCMap x, OutMap y, float const s = 1.f) const
  {
    auto const time = this->startForward(x, y, false);
    y.device(Threads::TensorDevice()) = x * scales.reshape(res).broadcast(brd) * x.constant(s);
    this->finishForward(y, time, false);
  }

  void iforward(InCMap x, OutMap y, float const s = 1.f) const
  {
    auto const time = this->startForward(x, y, true);
    y.device(Threads::TensorDevice()) += x * scales.reshape(res).broadcast(brd) * x.constant(s);
    this->finishForward(y, time, false);
  }

  void adjoint(OutCMap y, InMap x, float const s = 1.f) const
  {
    auto const time = this->startAdjoint(y, x, false);
    x.device(Threads::TensorDevice()) = y * scales.reshape(res).broadcast(brd) * y.constant(s);
    this->finishAdjoint(x, time, false);
  }

  void iadjoint(OutCMap y, InMap x, float const s = 1.f) const
  {
    auto const time = this->startAdjoint(y, x, true);
    x.device(Threads::TensorDevice()) += y * scales.reshape(res).broadcast(brd) * y.constant(s);
    this->finishAdjoint(x, time, false);
  }

  void inverse(OutCMap y, InMap x, float const s, float const b) const
  {
    auto const time = this->startInverse(y, x);
    x.device(Threads::TensorDevice()) = y / (scales.reshape(res).broadcast(brd) * y.constant(s) + y.constant(b));
    this->finishInverse(x, time);
  }

  auto operator+(Cx const s) const -> std::shared_ptr<Ops::Op>
  {
    TScales p = scales + s;
    return std::make_shared<TensorScale<Rank, FrontRank, BackRank>>(this->ishape, p);
  }

  auto weights() const -> TScales const & { return scales; }

private:
  TScales  scales;
  Sz<Rank> res, brd;
};

} // namespace rl::TOps
