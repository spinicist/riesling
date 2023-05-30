#pragma once

#include "types.hpp"

namespace rl {

struct Settings
{
  Index spokesPerSeg = 256, segsPerPrep = 2, segsPrep2 = 0, spokesSpoil = 0;
  float alpha = 1.f, ascale = 1.f, TR = 2.e-3f, Tramp = 10.e-3f, Tssi = 10.e-3f, TI = 0, Trec = 0, TE = 0, bval = 0;
  bool inversion;
};

struct Sequence
{
  Settings settings;

  Sequence(Settings const &s)
    : settings{s}
  {
  }

  virtual auto length() const -> Index = 0;
  virtual auto parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const
    -> Eigen::ArrayXXf = 0;
  virtual auto simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf = 0;
};

} // namespace rl
