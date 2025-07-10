#pragma once

#include "../types.hpp"

#include <chrono>

namespace rl::Ops {

struct Op
{
  using Vector = Eigen::Vector<Cx, Eigen::Dynamic>;
  using Map = Eigen::Map<Vector, Eigen::AlignedMax>;
  using CMap = Eigen::Map<Vector const, Eigen::AlignedMax>;
  using Ptr = std::shared_ptr<Op>;
  using Time = std::chrono::high_resolution_clock::time_point;

  std::string name;
  Op(std::string const &n);

  virtual auto rows() const -> Index = 0;
  virtual auto cols() const -> Index = 0;

  virtual void forward(CMap x, Map y, float const s = 1.f) const = 0;
  virtual void adjoint(CMap y, Map x, float const s = 1.f) const = 0;
  virtual void inverse(CMap y, Map x, float const s = 1.f, float const b = 0.f) const;
  virtual auto forward(Vector const &x, float const s = 1.f) const -> Vector;
  virtual auto adjoint(Vector const &y, float const s = 1.f) const -> Vector;
  void         forward(Vector const &x, Vector &y, float const s = 1.f) const;
  void         adjoint(Vector const &y, Vector &x, float const s = 1.f) const;
  void         inverse(Vector const &y, Vector &x, float const s = 1.f, float const b = 0.f) const;

  /* These versions scale and add in-place to the output */
  virtual void iforward(CMap x, Map y, float const s = 1.f) const = 0;
  virtual void iadjoint(CMap y, Map x, float const s = 1.f) const = 0;
  void         iforward(Vector const &x, Vector &y, float const s = 1.f) const;
  void         iadjoint(Vector const &y, Vector &x, float const s = 1.f) const;

protected:
  auto startForward(CMap x, Map const &y, bool const ip) const -> Time;
  void finishForward(Map const &y, Time const start, bool const ip) const;
  auto startAdjoint(CMap y, Map const &x, bool const ip) const -> Time;
  void finishAdjoint(Map const &x, Time const start, bool const ip) const;
  auto startInverse(CMap y, Map const &x) const -> Time;
  void finishInverse(Map const &x, Time const start) const;
};

#define OP_INHERIT                                                                                                             \
  using typename Op::Vector;                                                                                                   \
  using typename Op::Map;                                                                                                      \
  using typename Op::CMap;                                                                                                     \
  using typename Op::Ptr;                                                                                                      \
  using Op::forward;                                                                                                           \
  using Op::adjoint;                                                                                                           \
  auto rows() const -> Index final;                                                                                            \
  auto cols() const -> Index final;

} // namespace rl::Ops
