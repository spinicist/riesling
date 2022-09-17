#pragma once

#include "functor.hpp"

namespace rl {

struct DictionaryProjection final : Functor<Cx4>
{
  Re2 dictionary = Re2();
  DictionaryProjection(Re2);

  auto operator()(const Cx4 &) const -> Cx4;
};

} // namespace rl
