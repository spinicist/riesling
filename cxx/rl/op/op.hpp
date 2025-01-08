#pragma once

#include "../types.hpp"

#include <chrono>

namespace rl::Ops {

template <typename Scalar_ = Cx> struct Op
{
  using Scalar = Scalar_;
  using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using Map = typename Vector::AlignedMapType;
  using CMap = typename Vector::ConstAlignedMapType;
  using Ptr = std::shared_ptr<Op<Scalar>>;
  using Time = std::chrono::high_resolution_clock::time_point;

  std::string name;
  Op(std::string const &n);

  virtual auto rows() const -> Index = 0;
  virtual auto cols() const -> Index = 0;

  virtual void forward(CMap const &x, Map &y) const = 0;
  virtual void adjoint(CMap const &y, Map &x) const = 0;
  virtual void inverse(CMap const &y, Map &x) const;
  virtual auto forward(Vector const &x) const -> Vector;
  virtual auto adjoint(Vector const &y) const -> Vector;
  void         forward(Vector const &x, Vector &y) const;
  void         adjoint(Vector const &y, Vector &x) const;
  void         inverse(Vector const &y, Vector &x) const;

  /* These versions add in-place to the output */
  virtual void iforward(CMap const &x, Map &y) const = 0;
  virtual void iadjoint(CMap const &y, Map &x) const = 0;
  void         iforward(Vector const &x, Vector &y) const;
  void         iadjoint(Vector const &y, Vector &x) const;

  virtual auto inverse() const -> std::shared_ptr<Op<Scalar>>;
  virtual auto inverse(float const bias, float const scale) const -> std::shared_ptr<Op<Scalar>>;
  virtual auto operator+(Scalar const) const -> std::shared_ptr<Op<Scalar>>;

protected:
  auto startForward(CMap const &x, Map const &y, bool const ip) const -> Time;
  void finishForward(Map const &y, Time const start, bool const ip) const;
  auto startAdjoint(CMap const &y, Map const &x, bool const ip) const -> Time;
  void finishAdjoint(Map const &x, Time const start, bool const ip) const;
  auto startInverse(CMap const &y, Map const &x, bool const ip) const -> Time;
  void finishInverse(Map const &x, Time const start, bool const ip) const;
};

#define OP_INHERIT                                                                                                             \
  using typename Op<Scalar>::Vector;                                                                                           \
  using typename Op<Scalar>::Map;                                                                                              \
  using typename Op<Scalar>::CMap;                                                                                             \
  using Op<Scalar>::forward;                                                                                                   \
  using Op<Scalar>::adjoint;                                                                                                   \
  auto rows() const -> Index;                                                                                                  \
  auto cols() const -> Index;

} // namespace rl::Ops
