#pragma once

#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct T2FLAIR final : Sequence
{
  static const Index nParameters = 3;
  T2FLAIR(Settings const &s);

  auto length() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf;
};

} // namespace rl
