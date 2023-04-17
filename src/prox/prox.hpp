#pragma once

#include "types.hpp"

namespace rl {

template <typename Scalar = Cx>
struct Prox
{
  using Vector = Eigen::Vector<Scalar, Eigen::Dynamic>;

  virtual void operator()(float const λ, Vector const &x, Vector &z) const = 0;
  virtual auto operator()(float const λ, Vector const &x) const -> Vector {
    Vector z(x.size());
    this->operator()(λ, x, z);
    return z;
  }
  
  virtual ~Prox(){};
};

}
