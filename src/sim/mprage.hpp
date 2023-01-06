#pragma once

#include "log.hpp"
#include "parameter.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct MPRAGE final : Sequence
{
  MPRAGE(Settings const &s);

  auto length() const -> Index;
  auto parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const -> Eigen::ArrayXXf;
  auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf;
};

} // namespace rl
