#pragma once

#include "log.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct Prep final : Sequence
{
  Prep(Settings const &s);

  auto length() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Cx2;
};

struct Prep2 final : Sequence
{
  Prep2(Settings const &s);

  auto length() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Cx2;
};

} // namespace rl
