#pragma once

#include "functor.hpp"

namespace rl {

struct LLR final : Functor<Cx4>
{
  Index patchSize;
  float λ;
  bool sliding;
  LLR(float, Index, bool);

  auto operator()(Cx4 const &) const -> Cx4;

private:
  auto applySliding(Cx4 const &) const -> Cx4;
  auto applyFixed(Cx4 const &) const -> Cx4;
};

} // namespace rl
