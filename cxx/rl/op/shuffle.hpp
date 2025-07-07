#pragma once

#include "top-impl.hpp"
#include "compose.hpp"

#include "../log.hpp"

namespace rl::TOps {

template <typename S, int Rank> struct Shuffle final : TOp<S, Rank, Rank>
{
  TOP_INHERIT(S, Rank, Rank)
  using Parent::adjoint;
  using Parent::forward;
  using Ptr = std::shared_ptr<Shuffle>;

  Shuffle(Sz<Rank> const ishape, Sz<Rank> const is)
    : Parent("Shuffle", ishape, ishape)
    , ishuff{is}

  {
    Sz<Rank> osh;
    for (Index ii = 0; ii < Rank; ii++) {
      if (ishuff[ii] < 0 || ishuff[ii] > Rank - 1) { throw Log::Failure("Shuffle", "Invalid shuffle dimension {}", ishuff); }
      for (Index ij = 0; ij < Rank; ij++) { // Search for the shuffled dimension
        if (ishuff[ij] == ii) {
          oshuff[ii] = ij;
          break;
        }
      }
      oshape[ii] = ishape[ishuff[ii]];
    }
    Log::Print("Shuffle", "Input {} Output {}", ishuff, oshuff);
  }

  void forward(InCMap x, OutMap y) const
  {
    auto const time = this->startForward(x, y, false);
    y.device(Threads::TensorDevice()) = x.shuffle(ishuff);
    this->finishForward(y, time, false);
  }

  void adjoint(OutCMap y, InMap x) const
  {
    auto const                 time = this->startAdjoint(y, x, false);
    x.device(Threads::TensorDevice()) = y.shuffle(oshuff);
    this->finishAdjoint(x, time, false);
  }

  void iforward(InCMap x, OutMap y) const
  {
    auto const                time = this->startForward(x, y, true);
    y.device(Threads::TensorDevice()) += x.shuffle(ishuff);
    this->finishForward(y, time, true);
  }

  void iadjoint(OutCMap y, InMap x) const
  {
    auto const                 time = this->startAdjoint(y, x, true);
    x.device(Threads::TensorDevice()) += y.shuffle(oshuff);
    this->finishAdjoint(x, time, true);
  }

private:
  Sz<Rank> ishuff, oshuff;
};

template <int Rank> auto MakeShuffle(Sz<Rank> const ish, Sz<Rank> const ishuff)
  -> Shuffle<Cx, Rank>::Ptr
{
  return std::make_shared<Shuffle<Cx, Rank>>(ish, ishuff);
}

template <typename Op> auto MakeShuffleOutput(std::shared_ptr<Op> op, Sz<Op::OutRank> const shuff)
  -> Shuffle<Cx, Op::OutRank>::Ptr
{
  auto s = std::make_shared<Shuffle<Cx, Op::OutRank>>(op->oshape, shuff);
  return MakeCompose(op, s);
}

template <typename Op> auto MakeShuffleInput(std::shared_ptr<Op> op, Sz<Op::InRank> const shuff)
  -> TOps::TOp<Op::InRank, Op::OutRank>::Ptr
{
  auto s = std::make_shared<Shuffle<Cx, Op::InRank>>(op->ishape, shuff);
  return MakeCompose(s, op);
}

} // namespace rl::TOps
