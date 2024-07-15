#pragma once

#include "log.hpp"
#include "parameter.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct IR final : Sequence
{
  static const Index nParameters = 3;
  IR(Settings const &s);

  auto traces() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Cx2;
};

struct IR2 final : Sequence
{
  static const Index nParameters = 4;
  IR2(Settings const &s);

  auto traces() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Cx2;
};

} // namespace rl
