#pragma once

#include "op.hpp"

namespace rl::Ops {

struct Identity final : Op
{
  OP_INHERIT

  Identity(Index const s);
  static auto Make(Index const s) -> Ptr;
  void        forward(CMap x, Map y, float const s = 1.f) const;
  void        adjoint(CMap y, Map x, float const s = 1.f) const;
  void        inverse(CMap y, Map x, float const s = 1.f, float const b = 0.f) const;
  void        iforward(CMap x, Map y, float const s = 1.f) const;
  void        iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  Index sz;
};

struct MatMul final : Op
{
  OP_INHERIT
  using Matrix = Eigen::Matrix<Cx, Eigen::Dynamic, Eigen::Dynamic>;
  MatMul(Matrix const m);
  void forward(CMap x, Map y, float const s = 1.f) const;
  void adjoint(CMap y, Map x, float const s = 1.f) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  Matrix mat;
};

//! Scale the output of another Linear Operator
struct DiagScale final : Op
{
  OP_INHERIT
  DiagScale(Index const sz, float const s);
  static auto Make(Index const sz, float const s) -> Ptr;
  void        forward(CMap x, Map y, float const s = 1.f) const;
  void        adjoint(CMap y, Map x, float const s = 1.f) const;
  void        inverse(CMap y, Map x, float const s = 1.f, float const b = 0.f) const;
  void        iforward(CMap x, Map y, float const s = 1.f) const;
  void        iadjoint(CMap y, Map x, float const s = 1.f) const;

  float scale;

private:
  Index sz;
};

struct DiagRep final : Op
{
  OP_INHERIT
  DiagRep(Vector const &s, Index const repInner, Index const repOuter);
  void forward(CMap x, Map y, float const s = 1.f) const;
  void adjoint(CMap y, Map x, float const s = 1.f) const;
  void inverse(CMap y, Map x, float const s = 1.f, float const b = 0.f) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  Vector d;
  Index  rI, rO;
};

//! Multiply operators, i.e. y = A * B * x
struct Multiply final : Op
{
  OP_INHERIT
  Multiply(std::shared_ptr<Op> A, std::shared_ptr<Op> B);
  void forward(CMap x, Map y, float const s = 1.f) const;
  void adjoint(CMap y, Map x, float const s = 1.f) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  std::shared_ptr<Op> A, B;
  Vector mutable temp;
};

// Returns an Op representing A * B
auto Mul(typename Op::Ptr a, typename Op::Ptr b) -> typename Op::Ptr;

//! Vertically stack operators, i.e. A = [B; C]
struct VStack final : Op
{
  OP_INHERIT
  VStack(std::vector<Ptr> const &o);
  VStack(Ptr o1, std::vector<Ptr> const &o);
  static auto Make(std::vector<Ptr> const &o) -> Ptr;
  static auto Make(Ptr o1, std::vector<Ptr> const &o) -> Ptr;
  void        forward(CMap x, Map y, float const s = 1.f) const;
  void        adjoint(CMap y, Map x, float const s = 1.f) const;
  void        iforward(CMap x, Map y, float const s = 1.f) const;
  void        iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  void             check();
  std::vector<Ptr> ops;
};

//! Horizontally stack operators, i.e. A = [B C]
struct HStack final : Op
{
  OP_INHERIT
  HStack(std::vector<std::shared_ptr<Op>> const &o);
  HStack(std::shared_ptr<Op> op1, std::shared_ptr<Op> op2);
  HStack(std::shared_ptr<Op> op1, std::vector<std::shared_ptr<Op>> const &others);
  void forward(CMap x, Map y, float const s = 1.f) const;
  void adjoint(CMap y, Map x, float const s = 1.f) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  void                             check();
  std::vector<std::shared_ptr<Op>> ops;
};

//! Diagonally stack operators, i.e. A = [B 0; 0 C]
struct DStack final : Op
{
  OP_INHERIT
  DStack(std::vector<std::shared_ptr<Op>> const &o);
  DStack(std::shared_ptr<Op> op1, std::shared_ptr<Op> op2);
  void forward(CMap x, Map y, float const s = 1.f) const;
  void adjoint(CMap y, Map x, float const s = 1.f) const;
  void inverse(CMap y, Map x, float const s = 1.f, float const b = 0.f) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

  std::vector<std::shared_ptr<Op>> ops;
};

struct Extract final : Op
{
  OP_INHERIT
  Extract(Index const cols, Index const st, Index const rows);
  void forward(CMap x, Map y, float const s = 1.f) const;
  void adjoint(CMap y, Map x, float const s = 1.f) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  Index r, c, start;
};

struct Subtract final : Op
{
  OP_INHERIT
  Subtract(std::shared_ptr<Op> a, std::shared_ptr<Op> b);
  void forward(CMap x, Map y, float const s = 1.f) const;
  void adjoint(CMap y, Map x, float const s = 1.f) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  std::shared_ptr<Op> a, b;
};

// Returns an Op representing A - B
auto Sub(typename Op::Ptr a, typename Op::Ptr b) -> typename Op::Ptr;

} // namespace rl::Ops
