#pragma once

#include "log.hpp"
#include "parameter.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct T2Prep final : SegmentedZTE
{
  static const Index nParameters = 3;
  T2Prep(Parameters const &s);

  auto traces() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Sim;
};

} // namespace rl
