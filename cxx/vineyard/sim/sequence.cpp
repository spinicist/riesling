#include "sequence.hpp"

#include "log.hpp"

#include <fmt/format.h>

using namespace std::literals::complex_literals;

namespace rl {

std::unordered_map<std::string, Sequences> SequenceMap{
  {"Prep", Sequences::Prep}, {"Prep2", Sequences::Prep2},   {"IR", Sequences::IR},          {"IR2", Sequences::IR2},
  {"DIR", Sequences::DIR},   {"T2Prep", Sequences::T2Prep}, {"T2FLAIR", Sequences::T2FLAIR}};

auto Settings::format() const -> std::string
{
  return fmt::format(
    "Sequence settings: Samples {} Gap {} Spokes {} Spoil {} k0 {} Segs {} Segs2 {} Keep {} α {} α2 {} Tsamp {} TR {} "
    "Tramp {} Tssi {} TI {} Trec {} TE {}",
    samplesPerSpoke, samplesGap, spokesPerSeg, spokesSpoil, k0, segsPerPrep, segsPrep2, segsKeep, alpha, ascale, Tsamp, TR,
    Tramp, Tssi, TI, Trec, TE);
}

auto Sequence::offres(float const Δf) const -> Cx1
{
  if (settings.samplesPerSpoke < 1) {
    Cx1 o(1);
    o(0) = Cx(1.f);
    return o;
  }
  Cx1                       phase(settings.samplesPerSpoke);
  Eigen::VectorXcf::MapType pm(phase.data(), settings.samplesPerSpoke);
  float const               sampPhase = settings.Tsamp * Δf * 2 * M_PI;
  float const               startPhase = settings.samplesGap * sampPhase;
  float const               endPhase = (settings.samplesGap + settings.samplesPerSpoke - 1) * sampPhase;
  pm = Eigen::VectorXcf::LinSpaced(settings.samplesPerSpoke, startPhase * 1if, endPhase * 1if).array().exp();
  // Log::Print("Δf {} startPhase {} endPhase {}", Δf, startPhase, endPhase);
  return phase;
}

auto Sequence::readout(float const T2, float const Δf) const -> Cx1
{
  if (settings.samplesPerSpoke < 1) {
    Cx1 o(1);
    o(0) = Cx(1.f);
    return o;
  }

  float const               R2 = 1.f / T2;
  Cx1                       e(settings.samplesPerSpoke);
  Eigen::VectorXcf::MapType em(e.data(), settings.samplesPerSpoke);
  float const               sampPhase = settings.Tsamp * Δf * 2 * M_PI;
  float const               startPhase = settings.samplesGap * sampPhase;
  float const               endPhase = (settings.samplesGap + settings.samplesPerSpoke - 1) * sampPhase;
  float const               sampDecay = settings.Tsamp * R2;
  float const               startDecay = settings.samplesGap * sampDecay;
  float const               endDecay = (settings.samplesGap + settings.samplesPerSpoke - 1) * sampDecay;
  em = Eigen::VectorXcf::LinSpaced(settings.samplesPerSpoke, startPhase * 1if - startDecay, endPhase * 1if - endDecay)
         .array()
         .exp();
  // Log::Print("Δf {} startPhase {} endPhase {}", Δf, startPhase, endPhase);
  return e;
}

} // namespace rl