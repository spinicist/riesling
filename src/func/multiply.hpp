#pragma once

#include "functor.hpp"
#include "threads.hpp"
#include "tensorOps.hpp"
#include "log.hpp"

namespace rl {

template <typename Scalar, int Rank, int FrontRank = 1, int BackRank = 0>
struct BroadcastMultiply final : Functor<Eigen::Tensor<Scalar, Rank>>
{
  using Parent = Functor<Eigen::Tensor<Scalar, Rank>>;
  using typename Parent::Input;
  using typename Parent::Output;

  using Tensor = Eigen::Tensor<Scalar, Rank - FrontRank - BackRank>;
  Tensor a;
  std::string name;

  BroadcastMultiply(Tensor const &ain, std::string const &n = "BroadcastMultiply") : Parent(), a{ain}, name{n} {}
  void operator()(Input x, Output y) const
  {
    assert(x.dimensions() == y.dimensions());
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
    Log::Print<Log::Level::Debug>(FMT_STRING("{} dims {}->{} res {} brd {}"), name, x.dimensions(), y.dimensions(), res, brd);
    y.device(Threads::GlobalDevice()) = x * a.reshape(res).broadcast(brd);
  }
};

template <typename Scalar, int Rank, int FrontRank = 1, int BackRank = 0>
struct BroadcastPower final : Functor1<Eigen::Tensor<Scalar, Rank>>
{
  using Parent = Functor1<Eigen::Tensor<Scalar, Rank>>;
  using typename Parent::Input;
  using typename Parent::Output;

  using Tensor = Eigen::Tensor<Scalar, Rank - FrontRank - BackRank>;
  Tensor a;
  std::string name;

  BroadcastPower(Tensor const &ain, std::string const &n = "BroadcastPower") : Parent(), a{ain}, name{n} {}
  auto operator()(float const p, Input x) const -> Output
  {
    Output y(x.dimensions());
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
    Log::Print<Log::Level::Debug>(FMT_STRING("{} dims {}->{} res {} brd {}"), name, x.dimensions(), y.dimensions(), res, brd);
    y.device(Threads::GlobalDevice()) = x * a.pow(p).reshape(res).broadcast(brd);
    return y;
  }
};


} // namespace rl
