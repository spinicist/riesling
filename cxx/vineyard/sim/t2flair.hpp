#pragma once

#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct T2FLAIR final : Sequence
{
  static const Index nParameters = 4;
  T2FLAIR(Settings const &s);

  auto traces() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Cx2;
};

} // namespace rl
