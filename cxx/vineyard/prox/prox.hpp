#pragma once

#include "op/ops.hpp"
#include "types.hpp"

namespace rl::Proxs {

template <typename Scalar = Cx>
struct Prox
{
  using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using Map = Eigen::Map<Vector>;
  using CMap = Eigen::Map<Vector const>;
  using Op = Ops::Op<Scalar>;
  using Ptr = std::shared_ptr<Prox>;

  Prox(Index const sz);

  auto apply(float const α, Vector const &x) const -> Vector;
  auto apply(std::shared_ptr<Op> const α, Vector const &x) const -> Vector;

  void apply(float const α, Vector const &x, Vector &z) const;
  void apply(std::shared_ptr<Op> const α, Vector const &x, Vector &z) const;

  virtual void apply(float const α, CMap const &x, Map &z) const = 0;
  virtual void apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const;

  virtual ~Prox(){};

  Index sz;
};

#define PROX_INHERIT(Scalar)                                                                                                   \
  using Vector = typename Prox<Scalar>::Vector;                                                                                \
  using Map = typename Prox<Scalar>::Map;                                                                                      \
  using CMap = typename Prox<Scalar>::CMap;                                                                                    \
  using Op = typename Prox<Scalar>::Op;                                                                                        \
  using Prox<Scalar>::apply;

template <typename Scalar = Cx>
struct ConjugateProx final : Prox<Scalar>
{
  PROX_INHERIT(Scalar)

  ConjugateProx(std::shared_ptr<Prox<Scalar>> p);

  void apply(float const α, CMap const &x, Map &z) const;
  void apply(std::shared_ptr<Op> const α, CMap const &x, Map &z) const;

private:
  std::shared_ptr<Prox<Scalar>> p;
};

} // namespace rl::Proxs
