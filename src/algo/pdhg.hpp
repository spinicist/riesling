#pragma once

#include "op/ops.hpp"
#include "prox/l2.hpp"
#include "prox/prox.hpp"
#include "prox/stack.hpp"
#include "regularizers.hpp"

namespace rl {

struct PDHG
{
  using Op = Ops::Op<Cx>;
  using Prox = Proxs::Prox<Cx>;
  using Scalar = typename Op::Scalar;
  using Vector = typename Op::Vector;
  using CMap = typename Op::CMap;
  using Callback = std::function<void(Index const, Vector const &, Vector const &, Vector const &)>;

  PDHG(
    std::shared_ptr<Op> A,
    std::shared_ptr<Op> P,
    Regularizers const &reg,
    std::vector<float> const &σ = std::vector<float>(),
    float const τ = -1.f,
    Callback const &cb = nullptr);

  auto run(Cx const *bdata, Index const iterLimit) -> Vector;

  std::vector<float> σ;
  float τ;

private:
  std::shared_ptr<Op> Aʹ;
  std::shared_ptr<Proxs::L2<Cx>> l2;
  std::shared_ptr<Proxs::StackProx<Cx>> proxʹ;
  std::shared_ptr<Ops::Op<Cx>> σOp;
  Vector x, x̅, xold, xdiff, u, v;
  Callback debug = nullptr;
};

} // namespace rl
