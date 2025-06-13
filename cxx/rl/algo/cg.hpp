#pragma once

#include "../log/log.hpp"
#include "../op/ops.hpp"

namespace rl {
/*
 * Wrapper for solving normal equations
 */
template <typename Scalar_ = Cx> struct NormalOp final : Ops::Op<Scalar_>
{
  using Scalar = Scalar_;
  using Op = typename Ops::Op<Scalar>;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;
  using CMap = typename Op::CMap;

  std::shared_ptr<Op> op;
  NormalOp(std::shared_ptr<Op> o)
    : Op(o->name + " Normal")
    , op{o}
  {
  }

  auto rows() const -> Index { return op->cols(); }
  auto cols() const -> Index { return op->cols(); }

  auto forward(Vector const &x) const -> Vector { return op->adjoint(op->forward(x)); }
  auto adjoint(Vector const &y) const -> Vector { throw Log::Failure("CG", "Normal Operators do not have adjoints"); }

  void forward(CMap x, Map y) const
  {
    Vector temp(op->rows());
    Map    tm(temp.data(), temp.size());
    CMap   tcm(temp.data(), temp.size());
    op->forward(x, tm);
    op->adjoint(tcm, y);
  }

  void iforward(CMap x, Map y, float const s = 1.f) const
  {
    Vector temp(op->rows());
    Map    tm(temp.data(), temp.size());
    CMap   tcm(temp.data(), temp.size());
    op->forward(x, tm);
    op->iadjoint(tcm, y);
  }
  void adjoint(CMap x, Map y) const { throw Log::Failure("CG", "Normal Operators do not have adjoints"); }
  void iadjoint(CMap x, Map y) const { throw Log::Failure("CG", "Normal Operators do not have adjoints"); }
};

template <typename Op> auto MakeNormal(std::shared_ptr<Op> O) { return std::make_shared<NormalOp<typename Op::Scalar>>(O); }

struct ConjugateGradients
{
  using Op = Ops::Op<Cx>;
  using Vector = typename Op::Vector;
  using Map = typename Op::Map;
  using CMap = typename Op::CMap;

  struct Opts
  {
    Index imax = 4;
    float resTol = 1.e-6f;
  };

  Op::Ptr A;
  Opts    opts;

  auto run(Vector const &b, Vector const &x0 = Vector()) const -> Vector;
  auto run(CMap b, CMap x0 = CMap(nullptr, 0)) const -> Vector;
};

} // namespace rl
