#pragma once

#include "types.hpp"

namespace rl {

struct Settings
{
  Index samplesPerSpoke = 256, samplesGap = 2, spokesPerSeg = 256, spokesSpoil = 0, k0 = 0, segsPerPrep = 2, segsKeep = 2,
        segsPrep2 = 0;
  float alpha = 1.f, ascale = 1.f, Tsamp = 10e-6, TR = 2.e-3f, Tramp = 10.e-3f, Tssi = 10.e-3f, TI = 0, Trec = 0, TE = 0;

  auto format() const -> std::string;
};

struct Sequence
{
  Settings settings;

  Sequence(Settings const &s)
    : settings{s}
  {
  }

  virtual auto samples() const -> Index;
  virtual auto traces() const -> Index = 0;
  virtual auto simulate(Eigen::ArrayXf const &p) const -> Cx2 = 0;
  auto         offres(float const Δf) const -> Cx1;  // Off-resonance effects only
  auto         readout(float const T2, float const Δf) const -> Cx1; // Include transverse decay
};

enum struct Sequences
{
  NoPrep = 0,
  Prep,
  Prep2,
  IR,
  IR2,
  DIR,
  T2Prep,
  T2FLAIR
};

extern std::unordered_map<std::string, Sequences> SequenceMap;

} // namespace rl
