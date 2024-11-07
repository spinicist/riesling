#pragma once

#include "log.hpp"
#include "parameter.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct IR final : SegmentedZTE
{
  static const Index nParameters = 3;
  IR(Parameters const &s);

  auto simulate(Eigen::ArrayXf const &p) const -> Sim;
};

struct IR2 final : SegmentedZTE
{
  static const Index nParameters = 4;
  IR2(Parameters const &s);

  auto simulate(Eigen::ArrayXf const &p) const -> Sim;
};

} // namespace rl
