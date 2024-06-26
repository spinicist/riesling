#pragma once

#include "log.hpp"
#include "types.hpp"

namespace rl::Ops {

template <typename Scalar_ = Cx> struct Op
{
  using Scalar = Scalar_;
  using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using Map = typename Vector::AlignedMapType;
  using CMap = typename Vector::ConstAlignedMapType;
  using Ptr = std::shared_ptr<Op<Scalar>>;

  std::string name;
  Op(std::string const &n);

  virtual auto rows() const -> Index = 0;
  virtual auto cols() const -> Index = 0;
  
  virtual void forward(CMap const &x, Map &y) const = 0;
  virtual void adjoint(CMap const &y, Map &x) const = 0;
  virtual auto forward(Vector const &x) const -> Vector;
  virtual auto adjoint(Vector const &y) const -> Vector;
  void forward(Vector const &x, Vector &y) const;
  void adjoint(Vector const &y, Vector &x) const;

  /* These versions add in-place to the output */
  virtual void iforward(CMap const &x, Map &y) const = 0;
  virtual void iadjoint(CMap const &y, Map &x) const = 0;
  void iforward(Vector const &x, Vector &y) const;
  void iadjoint(Vector const &y, Vector &x) const;

  virtual auto inverse() const -> std::shared_ptr<Op<Scalar>>;
  virtual auto inverse(float const bias, float const scale) const -> std::shared_ptr<Op<Scalar>>;
  virtual auto operator+(Scalar const) const -> std::shared_ptr<Op<Scalar>>;
};

} // namespace rl::Ops
