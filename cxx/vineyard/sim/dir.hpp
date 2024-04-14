#pragma once

#include "log.hpp"
#include "parameter.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct DIR final : Sequence
{
  static const Index nParameters = 3;
  DIR(Settings const s);

  auto length() const -> Index;
  auto parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const -> Eigen::ArrayXXf;
  auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf;
};

struct Prep2 final : Sequence
{
  static const Index nParameters = 4;
  Prep2(Settings const s);

  auto length() const -> Index;
  auto parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const -> Eigen::ArrayXXf;
  auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf;
};

} // namespace rl
