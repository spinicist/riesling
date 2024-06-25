#pragma once

#include "log.hpp"
#include "parameter.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct IR final : Sequence
{
  static const Index nParameters = 1;
  IR(Settings const &s);

  auto length() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf;
};

struct IR2 final : Sequence
{
  static const Index nParameters = 2;
  IR2(Settings const &s);

  auto length() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf;
};

} // namespace rl
