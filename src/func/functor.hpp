#pragma once

#include "types.hpp"

/* Simple base class so we don't use std::function<T(const &T)> everywhere
 */

template <typename T>
struct Functor
{
  virtual auto operator()(T const &in) const -> T const & = 0;
  virtual ~Functor(){};
};

template <typename T>
struct Prox
{
  virtual auto operator()(float const λ, T const &in) const -> T const & = 0;
  virtual ~Prox(){};
};

template <typename T>
struct Identity final : Functor<T>
{
  auto operator()(T const &x) const -> T const & { return x; }
};
