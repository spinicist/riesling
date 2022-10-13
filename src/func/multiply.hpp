#pragma once

#include "functor.hpp"
#include "threads.hpp"
#include "tensorOps.hpp"

namespace rl {

template <typename Scalar, int Rank, int FrontRank = 1, int BackRank = 0>
struct BroadcastMultiply final : Functor<Eigen::Tensor<Scalar, Rank>>
{
  Eigen::Tensor<Scalar, Rank - FrontRank - BackRank> a;
  mutable Eigen::Tensor<Scalar, Rank> y;
  BroadcastMultiply(Eigen::Tensor<Scalar, Rank - FrontRank - BackRank> const &ain) : Functor<Eigen::Tensor<Scalar, Rank>>(), a{ain} {}
  auto operator()(Eigen::Tensor<Scalar, Rank> const &x) const -> Eigen::Tensor<Scalar, Rank> const &
  {
    assert(LastN<Rank - FrontRank - BackRank>(x.dimensions()) == a.dimensions());
    auto const start = Log::Now();
    y.resize(x.dimensions());
    Sz<Rank> res, brd;
    for (auto ii = 0; ii < FrontRank; ii++) {
      res[0] = 1;
      brd[0] = x.dimension(ii);
    }
    for (auto ii = FrontRank; ii < Rank - BackRank; ii++) {
      res[ii] = x.dimension(ii);
      brd[ii] = 1;
    }
    for (auto ii = Rank - BackRank; ii < Rank; ii++) {
      res[ii] = 1;
      brd[ii] = x.dimension(ii);
    }

    y.device(Threads::GlobalDevice()) = x * a.reshape(res).broadcast(brd);
    LOG_DEBUG(FMT_STRING("BroadcastMultiply Dims {} Norm {}->{}. Took {}"), y.dimensions(), Norm(x), Norm(y), Log::ToNow(start));
    return y;
  }
};

} // namespace rl
