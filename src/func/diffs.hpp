#pragma once

#include "functor.hpp"

namespace rl {

struct ForwardDiff final : Functor<Cx4>
{
  Index dim;
  ForwardDiff(Index);
  auto operator()(Cx4 const &) const -> Cx4 const &;
};

struct BackwardDiff final : Functor<Cx4>
{
  Index dim;
  BackwardDiff(Index);
  auto operator()(Cx4 const &) const -> Cx4 const &;
};

struct CentralDiff final : Functor<Cx4>
{
  Index dim;
  CentralDiff(Index);
  auto operator()(Cx4 const &) const -> Cx4 const &;
};

struct TotalDiff final : Functor<Cx4>
{
  TotalDiff();
  auto operator()(Cx4 const &) const -> Cx4 const &;

private:
  CentralDiff x_, y_, z_;
};

} // namespace rl
