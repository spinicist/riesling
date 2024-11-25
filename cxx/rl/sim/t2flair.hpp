#pragma once

#include "zte.hpp"

namespace rl {

struct T2FLAIR final : SegmentedZTE
{
  T2FLAIR(Pars const &s, bool const prepTraces);

  auto nTissueParameters() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Cx2;
  auto timepoints() const -> Re1;
};

} // namespace rl
