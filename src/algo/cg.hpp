#pragma once

#include "common.hpp"
#include "op/operator.hpp"
#include "signals.hpp"
#include "threads.hpp"

namespace rl {
/*
 * Wrapper for solving normal equations
 */
template <typename Scalar_ = Cx>
struct NormalOp final : Op::template Operator<Scalar_>
{
  using Scalar = Scalar_;
  using Op = typename Op::Operator<Scalar>;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;
  using CMap = typename Op::CMap;

  std::shared_ptr<Op> op;
  NormalOp(std::shared_ptr<Op> o)
    : Op(fmt::format("{} Normal", o->name))
    , op{o}
  {
  }

  auto rows() const -> Index { return op->cols(); }
  auto cols() const -> Index { return op->cols(); }

  auto forward(Vector const &x) const -> Vector { 
    return op->adjoint(op->forward(x)); }
  auto adjoint(Vector const &y) const -> Vector { Log::Fail("Normal Operators do not have adjoints"); }

  void forward(CMap const &x, Map &y) const
  {
    Vector temp(op->cols());
    Map tm(temp.data(), temp.size());
    CMap tcm(temp.data(), temp.size());
    op->forward(x, tm);
    op->adjoint(tcm, y);
  }
  void adjoint(CMap const &x, Map &y) const { Log::Fail("Normal Operators do not have adjoints"); }
};

struct ConjugateGradients
{
  using Op = Op::Operator<Cx>;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;

  std::shared_ptr<Op> op;
  Index iterLimit = 16;
  float resTol = 1.e-6f;
  bool debug = false;

  auto run(Cx *bdata, Cx *x0 = nullptr) const -> Vector;
};

} // namespace rl
