#pragma once
#include "log.hpp"
#include "operator.hpp"
#include "tensorOps.hpp"

namespace rl {

/* Scale input element-wise by factors, broadcasting as necessary.
 *
 * Equivalent to blockwise multiplication by a diagonal matrix
 * */
template <typename Scalar_, int Rank, int FrontRank = 1, int BackRank = 0>
struct Scale final : Operator<Scalar_, Rank, Rank>
{
  OP_INHERIT(Scalar_, Rank, Rank)
  using TScales = Eigen::Tensor<Scalar, Rank - FrontRank - BackRank>;

  Scale(InputDims const dims, TScales const &ain)
    : Parent("Scale", dims, dims)
    , a{ain}
  {
    for (auto ii = 0; ii < FrontRank; ii++) {
      res[0] = 1;
      brd[0] = dims[ii];
    }
    for (auto ii = FrontRank; ii < Rank - BackRank; ii++) {
      res[ii] = dims[ii];
      brd[ii] = 1;
    }
    for (auto ii = Rank - BackRank; ii < Rank; ii++) {
      res[ii] = 1;
      brd[ii] = dims[ii];
    }
    Log::Print<Log::Level::Debug>(FMT_STRING("{} dims {}->{} res {} brd {}"), this->name(), ain.dimensions(), dims, res, brd);
  }

  auto forward(InputMap x) const -> OutputMap
  {
    auto const time = this->startForward(x);
    x.device(Threads::GlobalDevice()) = x * a.reshape(res).broadcast(brd);
    this->finishForward(x, time);
    return x;
  }

  auto adjoint(OutputMap x) const -> InputMap
  {
    auto const time = this->startAdjoint(x);
    x.device(Threads::GlobalDevice()) = x * a.reshape(res).broadcast(brd);
    this->finishAdjoint(x, time);
    return x;
  }

private:
  TScales a;
  Sz<Rank> res, brd;
};

} // namespace rl
