#pragma once

#include "log.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct NoPrep final : Sequence
{
  static const Index nParameters = 1;
  NoPrep(Settings const &s);

  auto traces() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Cx2;
};

struct Prep final : Sequence
{
  static const Index nParameters = 3;
  Prep(Settings const &s);

  auto traces() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Cx2;
};

struct Prep2 final : Sequence
{
  static const Index nParameters = 4;
  Prep2(Settings const &s);

  auto traces() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Cx2;
};

} // namespace rl
