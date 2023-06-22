#pragma once

#include "types.hpp"
#include "op/ops.hpp"

namespace rl {

template <typename Scalar = Cx>
struct Prox
{
  using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using Map = Eigen::Map<Vector>;
  using CMap = Eigen::Map<Vector const>;

  Prox(Index const sz);

  void apply(float const α, Vector const &x, Vector &z) const;
  auto apply(float const α, Vector const &x) const -> Vector;
  virtual void apply(std::shared_ptr<Ops::Op<Scalar>> const α, CMap const &x, Map &z) const;
  virtual void apply(float const α, CMap const &x, Map &z) const = 0;

  virtual ~Prox(){};

  Index sz;
};

#define PROX_INHERIT(Scalar)                                                                                                   \
  using Prox<Scalar>::Vector;                                                                                                  \
  using Prox<Scalar>::Map;                                                                                                     \
  using Prox<Scalar>::CMap;

template <typename Scalar = Cx>
struct ConjugateProx final : Prox<Scalar>
{
  using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using Map = Eigen::Map<Vector>;
  using CMap = Eigen::Map<Vector const>;

  ConjugateProx(std::shared_ptr<Prox<Scalar>> p);

  void apply(float const α, CMap const &x, Map &z) const;

private:
  std::shared_ptr<Prox<Scalar>> p;
};

} // namespace rl
