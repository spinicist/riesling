#pragma once

#include "op.hpp"

namespace rl::Ops {

template <typename Scalar = Cx> struct Identity final : Op<Scalar>
{
  OP_INHERIT

  Identity(Index const s);
  static auto Make(Index const s) -> Identity::Ptr;
  void        forward(CMap x, Map y) const;
  void        adjoint(CMap y, Map x) const;
  void        inverse(CMap y, Map x, float const s = 1.f, float const b = 0.f) const;
  void        iforward(CMap x, Map y, float const s = 1.f) const;
  void        iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  Index sz;
};

template <typename Scalar = Cx> struct MatMul final : Op<Scalar>
{
  OP_INHERIT
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  MatMul(Matrix const m);
  void forward(CMap x, Map y) const;
  void adjoint(CMap y, Map x) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  Matrix mat;
};

//! Scale the output of another Linear Operator
template <typename Scalar = Cx> struct DiagScale final : Op<Scalar>
{
  OP_INHERIT
  DiagScale(Index const sz, float const s);
  static auto Make(Index const sz, float const s) -> DiagScale::Ptr;
  void        forward(CMap, Map) const;
  void        adjoint(CMap, Map) const;
  void        inverse(CMap y, Map x, float const s = 1.f, float const b = 0.f) const;
  void        iforward(CMap x, Map y, float const s = 1.f) const;
  void        iadjoint(CMap y, Map x, float const s = 1.f) const;

  float scale;

private:
  Index sz;
};

template <typename Scalar = Cx> struct DiagRep final : Op<Scalar>
{
  OP_INHERIT
  DiagRep(Vector const &s, Index const repInner, Index const repOuter);
  void forward(CMap x, Map y) const;
  void adjoint(CMap y, Map x) const;
  void inverse(CMap y, Map x, float const s = 1.f, float const b = 0.f) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  Vector d;
  Index  rI, rO;
};

//! Multiply operators, i.e. y = A * B * x
template <typename Scalar = Cx> struct Multiply final : Op<Scalar>
{
  OP_INHERIT
  Multiply(std::shared_ptr<Op<Scalar>> A, std::shared_ptr<Op<Scalar>> B);
  void forward(CMap x, Map y) const;
  void adjoint(CMap y, Map x) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  std::shared_ptr<Op<Scalar>> A, B;
  Vector mutable temp;
};

// Returns an Op representing A * B
template <typename S = Cx> auto Mul(typename Op<S>::Ptr a, typename Op<S>::Ptr b) -> typename Op<S>::Ptr;

//! Vertically stack operators, i.e. A = [B; C]
template <typename Scalar = Cx> struct VStack final : Op<Scalar>
{
  OP_INHERIT
  VStack(std::vector<std::shared_ptr<Op<Scalar>>> const &o);
  VStack(std::shared_ptr<Op<Scalar>> op1, std::shared_ptr<Op<Scalar>> op2);
  VStack(std::shared_ptr<Op<Scalar>> op1, std::vector<std::shared_ptr<Op<Scalar>>> const &others);
  void forward(CMap x, Map y) const;
  void adjoint(CMap y, Map x) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  void                                     check();
  std::vector<std::shared_ptr<Op<Scalar>>> ops;
};

//! Horizontally stack operators, i.e. A = [B C]
template <typename Scalar = Cx> struct HStack final : Op<Scalar>
{
  OP_INHERIT
  HStack(std::vector<std::shared_ptr<Op<Scalar>>> const &o);
  HStack(std::shared_ptr<Op<Scalar>> op1, std::shared_ptr<Op<Scalar>> op2);
  HStack(std::shared_ptr<Op<Scalar>> op1, std::vector<std::shared_ptr<Op<Scalar>>> const &others);
  void forward(CMap x, Map y) const;
  void adjoint(CMap y, Map x) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  void                                     check();
  std::vector<std::shared_ptr<Op<Scalar>>> ops;
};

//! Diagonally stack operators, i.e. A = [B 0; 0 C]
template <typename Scalar = Cx> struct DStack final : Op<Scalar>
{
  OP_INHERIT
  DStack(std::vector<std::shared_ptr<Op<Scalar>>> const &o);
  DStack(std::shared_ptr<Op<Scalar>> op1, std::shared_ptr<Op<Scalar>> op2);
  void forward(CMap x, Map y) const;
  void adjoint(CMap y, Map x) const;
  void inverse(CMap y, Map x, float const s = 1.f, float const b = 0.f) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

  std::vector<std::shared_ptr<Op<Scalar>>> ops;
};

template <typename Scalar = Cx> struct Extract final : Op<Scalar>
{
  OP_INHERIT
  Extract(Index const cols, Index const st, Index const rows);
  void forward(CMap x, Map y) const;
  void adjoint(CMap y, Map x) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  Index r, c, start;
};

template <typename Scalar = Cx> struct Subtract final : Op<Scalar>
{
  OP_INHERIT
  Subtract(std::shared_ptr<Op<Scalar>> a, std::shared_ptr<Op<Scalar>> b);
  void forward(CMap x, Map y) const;
  void adjoint(CMap y, Map x) const;
  void iforward(CMap x, Map y, float const s = 1.f) const;
  void iadjoint(CMap y, Map x, float const s = 1.f) const;

private:
  std::shared_ptr<Op<Scalar>> a, b;
};

// Returns an Op representing A - B
template <typename S = Cx> auto Sub(typename Op<S>::Ptr a, typename Op<S>::Ptr b) -> typename Op<S>::Ptr;

} // namespace rl::Ops
