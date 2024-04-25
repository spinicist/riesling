#pragma once

#include "log.hpp"
#include "sequence.hpp"
#include "types.hpp"

namespace rl {

struct GenPrep final : Sequence
{
  static const Index nParameters = 2;
  GenPrep(Settings const &s);

  auto length() const -> Index;
  auto parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const -> Eigen::ArrayXXf;
  auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf;
};

struct GenPrep2 final : Sequence
{
  static const Index nParameters = 3;
  GenPrep2(Settings const &s);

  auto length() const -> Index;
  auto parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const -> Eigen::ArrayXXf;
  auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf;
};

} // namespace rl
