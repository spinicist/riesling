#pragma once

#include "types.hpp"

namespace rl {

template <typename Scalar = Cx>
struct Prox
{
  using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;
  using Map = Eigen::Map<Vector>;
  using CMap = Eigen::Map<Vector const>;

  void apply(float const α, Vector const &x, Vector &z) const
  {
    CMap xm(x.data(), x.size());
    Map zm(z.data(), z.size());
    this->apply(α, xm, zm);
  }
  auto apply(float const α, Vector const &x) const -> Vector
  {
    Vector z(x.size());
    this->apply(α, x, z);
    return z;
  }
  virtual void apply(float const α, CMap const &x, Map &z) const = 0;

  virtual ~Prox(){};
};

#define PROX_INHERIT(Scalar)                                                                                                   \
  using Prox<Scalar>::Vector;                                                                                                  \
  using Prox<Scalar>::Map;                                                                                                     \
  using Prox<Scalar>::CMap;

} // namespace rl
