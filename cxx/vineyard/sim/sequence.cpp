#include "sequence.hpp"

#include "log.hpp"

#include <fmt/format.h>

using namespace std::literals::complex_literals;

namespace rl {

std::unordered_map<std::string, Sequences> SequenceMap{
  // {"IR", Sequences::IR},           {"IR2", Sequences::IR2},
  {"DIR", Sequences::DIR},        // {"T2Prep", Sequences::T2Prep},
  {"T2FLAIR", Sequences::T2FLAIR} //, {"Prep", Sequences::GenPrep}, {"Prep2", Sequences::GenPrep2}
};


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
  Cx1                       phase(settings.samplesPerSpoke);
  Eigen::VectorXcf::MapType pm(phase.data(), settings.samplesPerSpoke);
  float const               sampPhase = settings.Tsamp * Δf * 2 * M_PI;
  float const               startPhase = settings.samplesGap * sampPhase;
  float const               endPhase = (settings.samplesGap + settings.samplesPerSpoke - 1) * sampPhase;
  pm = Eigen::VectorXcf::LinSpaced(settings.samplesPerSpoke, startPhase * 1if, endPhase * 1if).array().exp();
  // Log::Print("Δf {} startPhase {} endPhase {}", Δf, startPhase, endPhase);
  return phase;
}

} // namespace rl