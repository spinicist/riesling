#pragma once

#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct T2FLAIR final : SegmentedZTE
{
  static const Index nParameters = 4;
  T2FLAIR(Parameters const &s, bool const prepTraces);

  auto simulate(Eigen::ArrayXf const &p) const -> Sim;
};

} // namespace rl
