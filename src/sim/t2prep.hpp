#pragma once

#include "log.hpp"
#include "parameter.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct T2Prep final : Sequence
{
  static const Index nParameters = 3;
  T2Prep(Settings const &s);

  auto length() const -> Index;
  auto parameters(Index const nsamp, std::vector<float> const lo, std::vector<float> const hi) const -> Eigen::ArrayXXf;
  auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf;
};

} // namespace rl
