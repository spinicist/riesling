#pragma once

#include "op.hpp"

namespace rl::Ops {

template <typename Scalar = Cx> struct Identity final : Op<Scalar>
{
  OP_INHERIT
  Identity(Index const s);
  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;
  void iforward(CMap const &x, Map &y) const;
  void iadjoint(CMap const &y, Map &x) const;
private:
  Index sz;
};

template <typename Scalar = Cx> struct MatMul final : Op<Scalar>
{
  OP_INHERIT
  using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  MatMul(Matrix const m);
  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;
  void iforward(CMap const &x, Map &y) const;
  void iadjoint(CMap const &y, Map &x) const;
private:
  Matrix mat;
};

//! Scale the output of another Linear Operator
template <typename Scalar = Cx> struct DiagScale final : Op<Scalar>
{
  OP_INHERIT
  DiagScale(Index const sz, float const s);
  auto  inverse() const -> std::shared_ptr<Op<Scalar>>;
  void forward(CMap const &, Map &) const;
  void adjoint(CMap const &, Map &) const;
  void iforward(CMap const &x, Map &y) const;
  void iadjoint(CMap const &y, Map &x) const;
  float scale;
private:
  Index sz;
};

template <typename Scalar = Cx> struct DiagRep final : Op<Scalar>
{
  OP_INHERIT
  DiagRep(Index const reps, Vector const &s);
  DiagRep(Index const reps, Vector const &s, float const b, float const sc);
  auto inverse() const -> std::shared_ptr<Op<Scalar>> final;
  auto inverse(float const bias, float const scale) const -> std::shared_ptr<Op<Scalar>> final;
  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;
  void iforward(CMap const &x, Map &y) const;
  void iadjoint(CMap const &y, Map &x) const;
private:
  Index  reps;
  Vector s;
  bool   isInverse = false;
  float  bias = 0.f, scale = 0.f;
};

//! Multiply operators, i.e. y = A * B * x
template <typename Scalar = Cx> struct Multiply final : Op<Scalar>
{
  OP_INHERIT
  Multiply(std::shared_ptr<Op<Scalar>> A, std::shared_ptr<Op<Scalar>> B);
  auto inverse() const -> std::shared_ptr<Op<Scalar>> final;
  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;
  void iforward(CMap const &x, Map &y) const;
  void iadjoint(CMap const &y, Map &x) const;
private:
  std::shared_ptr<Op<Scalar>> A, B;
};

//! Vertically stack operators, i.e. A = [B; C]
template <typename Scalar = Cx> struct VStack final : Op<Scalar>
{
  OP_INHERIT
  VStack(std::vector<std::shared_ptr<Op<Scalar>>> const &o);
  VStack(std::shared_ptr<Op<Scalar>> op1, std::shared_ptr<Op<Scalar>> op2);
  VStack(std::shared_ptr<Op<Scalar>> op1, std::vector<std::shared_ptr<Op<Scalar>>> const &others);
  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;
  void iforward(CMap const &x, Map &y) const;
  void iadjoint(CMap const &y, Map &x) const;
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
  auto inverse() const -> std::shared_ptr<Op<Scalar>> final;
  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;
  void iforward(CMap const &x, Map &y) const;
  void iadjoint(CMap const &y, Map &x) const;
  std::vector<std::shared_ptr<Op<Scalar>>> ops;
};

template <typename Scalar = Cx> struct Extract final : Op<Scalar>
{
  OP_INHERIT
  Extract(Index const cols, Index const st, Index const rows);
  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;
  void iforward(CMap const &x, Map &y) const;
  void iadjoint(CMap const &y, Map &x) const;
private:
  Index r, c, start;
};

template <typename Scalar = Cx> struct Subtract final : Op<Scalar>
{
  OP_INHERIT
  Subtract(std::shared_ptr<Op<Scalar>> a, std::shared_ptr<Op<Scalar>> b);
  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;
  void iforward(CMap const &x, Map &y) const;
  void iadjoint(CMap const &y, Map &x) const;
private:
  std::shared_ptr<Op<Scalar>> a, b;
};

} // namespace rl::Ops
