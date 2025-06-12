#pragma once

#include "../op/ops.hpp"
#include "../prox/lsq.hpp"
#include "../prox/prox.hpp"
#include "../prox/stack.hpp"
#include "regularizer.hpp"

namespace rl {

struct PDHG
{
  using Op = Ops::Op<Cx>;
  using Prox = Proxs::Prox<Cx>;
  using Scalar = typename Op::Scalar;
  using Vector = typename Op::Vector;
  using CMap = typename Op::CMap;
  using Callback = std::function<void(Index const, Vector const &, Vector const &, Vector const &)>;

  PDHG(std::shared_ptr<Op>             A,
       std::shared_ptr<Op>             P,
       std::vector<Regularizer> const &regs,
       std::vector<float> const       &σ = std::vector<float>(),
       float const                     τ = -1.f,
       Callback const                 &cb = nullptr);

  auto run(Vector const &b) const -> Vector;
  auto run(CMap b) const -> Vector;

  std::vector<float> σ;
  float              τ;
  Index              imax = 4;

private:
  Op::Ptr Aʹ;
  Proxs::LeastSquares<Cx>::Ptr l2;
  Proxs::StackProx<Cx>::Ptr proxʹ;
  Op::Ptr                   σOp;
  Callback                  debug = nullptr;
};

} // namespace rl
