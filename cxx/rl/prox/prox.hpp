#pragma once

#include "../op/ops.hpp"
#include "../types.hpp"

namespace rl::Proxs {

template <typename Scalar = Cx> struct Prox
{
  using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using Map = typename Vector::AlignedMapType;
  using CMap = typename Eigen::Map<Vector const, Eigen::AlignedMax>;
  using Op = Ops::Op<Scalar>;
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

#define PROX_INHERIT(Scalar)                                                                                                   \
  using Vector = typename Prox<Scalar>::Vector;                                                                                \
  using Map = typename Prox<Scalar>::Map;                                                                                      \
  using CMap = typename Prox<Scalar>::CMap;                                                                                    \
  using Op = typename Prox<Scalar>::Op;                                                                                        \
  using Prox<Scalar>::primal;                                                                                                  \
  using Prox<Scalar>::dual;

} // namespace rl::Proxs
