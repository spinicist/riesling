#pragma once

#include "../op/ops.hpp"
#include "../types.hpp"

namespace rl::Proxs {

struct Prox
{
  using Vector = Eigen::Vector<Cx, Eigen::Dynamic>;
  using Map = typename Vector::AlignedMapType;
  using CMap = typename Eigen::Map<Vector const, Eigen::AlignedMax>;
  using Op = Ops::Op;
  using Ptr = std::shared_ptr<Prox>;

  Prox(Index const sz);

  auto         primal(float const α, Vector const &x) const -> Vector;
  void         primal(float const α, Vector const &x, Vector &z) const;
  virtual void primal(float const α, CMap x, Map z) const = 0;

  auto         dual(float const α, Vector const &x) const -> Vector;
  void         dual(float const α, Vector const &x, Vector &z) const;
  virtual void dual(float const α, CMap x, Map z) const = 0;

  virtual ~Prox() {};

  Index sz;
};

#define PROX_INHERIT                                                                                                           \
  using Vector = typename Prox::Vector;                                                                                        \
  using Map = typename Prox::Map;                                                                                              \
  using CMap = typename Prox::CMap;                                                                                            \
  using Op = typename Prox::Op;                                                                                                \
  using Ptr = Prox::Ptr;                                                                                                       \
  using Prox::primal;                                                                                                          \
  using Prox::dual;

} // namespace rl::Proxs
