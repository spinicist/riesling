#pragma once

#include "functor.hpp"
#include "threads.hpp"
#include "tensorOps.hpp"

namespace rl {

template <typename Scalar, int Rank, int FrontRank = 1, int BackRank = 0>
struct BroadcastMultiply final : Functor<Eigen::Tensor<Scalar, Rank>>
{
  using FullTensor = Eigen::Tensor<Scalar, Rank>;
  using ReducedTensor = Eigen::Tensor<Scalar, Rank - FrontRank - BackRank>;
  ReducedTensor a;

  BroadcastMultiply(ReducedTensor const &ain) : Functor<FullTensor>(), a{ain} {}
  auto operator()(Eigen::TensorMap<FullTensor const> x) const -> Eigen::TensorMap<FullTensor>
  {
    assert(LastN<Rank - FrontRank - BackRank>(x.dimensions()) == a.dimensions());
    static FullTensor y(x.dimensions());
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
    return y;
  }
};

} // namespace rl
