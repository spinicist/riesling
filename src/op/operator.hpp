#pragma once

#include "log.hpp"
#include "types.hpp"

#include <initializer_list>

namespace rl {

namespace LinOps {

template <typename Scalar_ = Cx>
struct Op
{
  using Scalar = Scalar_;
  using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using Map = Eigen::Map<Vector>;
  using CMap = Eigen::Map<Vector const>;

  std::string name;
  Op(std::string const &n);

  virtual auto rows() const -> Index = 0;
  virtual auto cols() const -> Index = 0;
  virtual void forward(CMap const &x, Map &y) const = 0;
  virtual void adjoint(CMap const &x, Map &y) const = 0;
  virtual auto forward(Vector const &x) const -> Vector;
  virtual auto adjoint(Vector const &y) const -> Vector;
  void forward(Vector const &x, Vector &y) const;
  void adjoint(Vector const &y, Vector &x) const;
};

template <typename Scalar = Cx>
struct Identity final : Op<Scalar>
{
  using typename Op<Scalar>::Map;
  using typename Op<Scalar>::CMap;

  Identity(Index const s);

  auto rows() const -> Index;
  auto cols() const -> Index;
  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;

private:
  Index sz;
};

//! Scale the output of another Linear Operator
template <typename Scalar = Cx>
struct Scale final : Op<Scalar>
{
  using typename Op<Scalar>::Map;
  using typename Op<Scalar>::CMap;

  Scale(std::shared_ptr<Op<Scalar>> op, float const s);

  auto rows() const -> Index;
  auto cols() const -> Index;

  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;

  std::shared_ptr<Op<Scalar>> op;
  float scale;
};

//! Concatenate operators, i.e. A * B
template <typename Scalar = Cx>
struct Concat final : Op<Scalar>
{
  using typename Op<Scalar>::Vector;
  using typename Op<Scalar>::Map;
  using typename Op<Scalar>::CMap;

  std::shared_ptr<Op<Scalar>> a, b;

  Concat(std::shared_ptr<Op<Scalar>> a_, std::shared_ptr<Op<Scalar>> b_);

  auto rows() const -> Index;
  auto cols() const -> Index;

  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;
};

//! Vertically stack operators, i.e. A = [B; C]
template <typename Scalar = Cx>
struct VStack final : Op<Scalar>
{
  using typename Op<Scalar>::Vector;
  using typename Op<Scalar>::Map;
  using typename Op<Scalar>::CMap;

  VStack(std::vector<std::shared_ptr<Op<Scalar>>> const &o);
  VStack(std::shared_ptr<Op<Scalar>> op1, std::shared_ptr<Op<Scalar>> op2);
  auto rows() const -> Index;
  auto cols() const -> Index;

  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;

private:
  void check();
  std::vector<std::shared_ptr<Op<Scalar>>> ops;
};

//! Diagonally stack operators, i.e. A = [B 0; 0 C]
template <typename Scalar = Cx>
struct DStack final : Op<Scalar>
{
  using typename Op<Scalar>::Vector;
  using typename Op<Scalar>::Map;
  using typename Op<Scalar>::CMap;

  DStack(std::vector<std::shared_ptr<Op<Scalar>>> const &o);
  DStack(std::shared_ptr<Op<Scalar>> op1, std::shared_ptr<Op<Scalar>> op2);
  auto rows() const -> Index;
  auto cols() const -> Index;

  void forward(CMap const &x, Map &y) const;
  void adjoint(CMap const &y, Map &x) const;

private:
  std::vector<std::shared_ptr<Op<Scalar>>> ops;
};

}

} // namespace rl
