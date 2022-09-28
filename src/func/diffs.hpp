#pragma once

#include "functor.hpp"

namespace rl {

struct ForwardDiff final : Functor<Cx4>
{
  Index dim;
  ForwardDiff(Index);
  auto operator()(Cx4 const &) const -> Cx4;
};

struct BackwardDiff final : Functor<Cx4>
{
  Index dim;
  BackwardDiff(Index);
  auto operator()(Cx4 const &) const -> Cx4;
};

struct CentralDiff final : Functor<Cx4>
{
  Index dim;
  CentralDiff(Index);
  auto operator()(Cx4 const &) const -> Cx4;
};

struct TotalDiff final : Functor<Cx4>
{
  TotalDiff();
  auto operator()(Cx4 const &) const -> Cx4;

private:
  CentralDiff x_, y_, z_;
};

} // namespace rl
