#pragma once

#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct T2FLAIR final : Sequence
{
  T2FLAIR(Settings const &s);

  auto length() const -> Index;
  auto parameters(Index const nsamp) const -> Eigen::ArrayXXf;
  auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf;
};

struct MultiFLAIR final : Sequence
{
  MultiFLAIR(Settings const &s);

  auto length() const -> Index;
  auto parameters(Index const nsamp) const -> Eigen::ArrayXXf;
  auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf;
};

} // namespace rl
