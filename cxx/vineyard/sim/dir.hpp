#pragma once

#include "log.hpp"
#include "parameter.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct DIR final : SegmentedZTE
{
  static const Index nParameters = 4;
  DIR(Parameters const s);

  auto traces() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Sim;
};

} // namespace rl
