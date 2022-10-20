#pragma once

#include "functor.hpp"

namespace rl {

struct ForwardDiff final : Functor<Cx4>
{
  using Parent = Functor<Cx4>;
  using typename Parent::Input;
  using typename Parent::Output;

  Index dim;
  ForwardDiff(Index);
  void operator()(Input x, Output y) const;
};

struct BackwardDiff final : Functor<Cx4>
{
    using Parent = Functor<Cx4>;
  using typename Parent::Input;
  using typename Parent::Output;

  Index dim;
  BackwardDiff(Index);
  void operator()(Input x, Output y) const;
};

struct CentralDiff final : Functor<Cx4>
{
  using Parent = Functor<Cx4>;
  using typename Parent::Input;
  using typename Parent::Output;

  Index dim;
  CentralDiff(Index);
  void operator()(Input x, Output y) const;
};

} // namespace rl
