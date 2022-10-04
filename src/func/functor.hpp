#pragma once

#include "types.hpp"

/* Simple base class so we don't use std::function<T(const &T)> everywhere
 */

template <typename T>
struct Functor
{
  virtual auto operator()(T const &in) const -> T = 0;
  virtual ~Functor() {};
};

template <typename T>
struct Prox
{
  virtual auto operator()(float const λ, T const &in) const -> T = 0;
  virtual ~Prox() {};
};