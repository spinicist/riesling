#pragma once

#include "log.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct NoPrep final : SegmentedZTE
{
  static const Index nParameters = 1;
  NoPrep(Parameters const &s);

  auto traces() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Sim;
};

struct Prep final : SegmentedZTE
{
  static const Index nParameters = 3;
  Prep(Parameters const &s);

  auto traces() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Sim;
};

struct Prep2 final : SegmentedZTE
{
  static const Index nParameters = 4;
  Prep2(Parameters const &s);

  auto traces() const -> Index;
  auto simulate(Eigen::ArrayXf const &p) const -> Sim;
};

} // namespace rl
