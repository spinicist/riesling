#pragma once

#include "types.hpp"

/* Simple base class so we don't use std::function<T(const &T)> everywhere
 */

template <typename T>
struct Functor
{
  using Input = Eigen::TensorMap<T const>;
  using Output = Eigen::TensorMap<T>;
  virtual void operator()(Input x, Output y) const = 0;
  virtual ~Functor(){};
};

template <typename T>
struct IdentityFunctor final : Functor<T>
{
  using typename Functor<T>::Input;
  using typename Functor<T>::Output;
  void operator()(Input x, Output y) const { y = x; }
};

template <typename T>
struct Functor1
{
  using Input = Eigen::TensorMap<T const>;
  using Output = T;

  virtual auto operator()(float const λ, Input in) const -> T = 0;
  virtual ~Functor1(){};
};

template<typename T>
using Prox = Functor1<T>;

template <typename T>
struct IdentityProx final : Prox<T>
{
  auto operator()(float const λ, Eigen::TensorMap<T const> in) const -> T { return in; }
};
