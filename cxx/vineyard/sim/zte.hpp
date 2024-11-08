#pragma once

#include "types.hpp"

namespace rl {

struct SegmentedZTE
{
  struct Pars
  {
    Index samplesPerSpoke = 256, samplesGap = 2, spokesPerSeg = 256, spokesSpoil = 0, k0 = 0, segsPerPrep = 2, segsPrep2 = 0;
    float alpha = 1.f, ascale = 1.f, Tsamp = 10e-6, TR = 2.e-3f, Tramp = 10.e-3f, Tssi = 10.e-3f, TI = 0, Trec = 0, TE = 0;

    auto format() const -> std::string;
  };
  struct Sim
  {
    Cx2 Mxy;
    Re1 t;
  };

  Pars p;
  bool       preseg; // Add a fake pre-segment trace

  SegmentedZTE(Pars const &s, bool const ps)
    : p{s}
    , preseg{ps}
  {
  }

  virtual auto samples() const -> Index;
  virtual auto traces() const -> Index;
  virtual auto nTissueParameters() const -> Index;
  virtual auto simulate(Eigen::ArrayXf const &p) const -> Cx2;
  virtual auto timepoints() const -> Re1;
  auto         offres(float const Δf) const -> Cx1;                  // Off-resonance effects only
  auto         readout(float const T2, float const Δf) const -> Cx1; // Include transverse decay

protected:
  auto E(float const R, float const t) const -> Eigen::Matrix2f;
  auto ET2p(float const R2, float const t) const -> Eigen::Matrix2f;
  auto A(float const a) const -> Eigen::Matrix2f;
  auto Eseg(float const R1, float const B1) const -> Eigen::Matrix2f;
  auto inv(float const Q) const -> Eigen::Matrix2f;
  void segment(Index &tp, Eigen::Vector2f &Mz, Cx1 &s0, float const R1, float const B1) const;
  void segmentTimepoints(Index &tp, float &t, Re1 &ts) const;
};

enum struct Sequences
{
  ZTE = 0,
  IR,
  DIR,
  T2Prep,
  T2FLAIR
};

extern std::unordered_map<std::string, Sequences> SequenceMap;

} // namespace rl
