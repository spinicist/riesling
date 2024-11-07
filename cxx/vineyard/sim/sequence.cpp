#include "sequence.hpp"

#include "log.hpp"

#include <fmt/format.h>
#include "unsupported/Eigen/MatrixFunctions"

using namespace std::literals::complex_literals;

namespace rl {

std::unordered_map<std::string, Sequences> SequenceMap{{"ZTE", Sequences::ZTE},       {"IR", Sequences::IR},
                                                       {"IR2", Sequences::IR2},       {"DIR", Sequences::DIR},
                                                       {"T2Prep", Sequences::T2Prep}, {"T2FLAIR", Sequences::T2FLAIR}};

auto SegmentedZTE::Parameters::format() const -> std::string
{
  return fmt::format("SegmentedZTE p: Samples {} Gap {} Spokes {} Spoil {} k0 {} Segs {} Segs2 {} α {} α2 {} Tsamp {} TR {} "
                     "Tramp {} Tssi {} TI {} Trec {} TE {}",
                     samplesPerSpoke, samplesGap, spokesPerSeg, spokesSpoil, k0, segsPerPrep, segsPrep2, alpha, ascale, Tsamp,
                     TR, Tramp, Tssi, TI, Trec, TE);
}

auto SegmentedZTE::samples() const -> Index { return std::max(1L, p.samplesPerSpoke); }
auto SegmentedZTE::traces() const -> Index { return (p.spokesPerSeg + p.k0 + (preseg ? 1 : 0)) * p.segsPerPrep; }

auto SegmentedZTE::offres(float const Δf) const -> Cx1
{
  if (p.samplesPerSpoke < 1) {
    Cx1 o(1);
    o(0) = Cx(1.f);
    return o;
  }
  Cx1                       phase(p.samplesPerSpoke);
  Eigen::VectorXcf::MapType pm(phase.data(), p.samplesPerSpoke);
  float const               sampPhase = p.Tsamp * Δf * 2 * M_PI;
  float const               startPhase = p.samplesGap * sampPhase;
  float const               endPhase = (p.samplesGap + p.samplesPerSpoke - 1) * sampPhase;
  pm = Eigen::VectorXcf::LinSpaced(p.samplesPerSpoke, startPhase * 1if, endPhase * 1if).array().exp();
  // Log::Print("Δf {} startPhase {} endPhase {}\n{}\n", Δf, startPhase, endPhase, fmt::streamed(pm.transpose()));
  return phase;
}

auto SegmentedZTE::readout(float const T2, float const Δf) const -> Cx1
{
  if (p.samplesPerSpoke < 1) {
    Cx1 o(1);
    o(0) = Cx(1.f);
    return o;
  }

  float const               R2 = 1.f / T2;
  Cx1                       e(p.samplesPerSpoke);
  Eigen::VectorXcf::MapType em(e.data(), p.samplesPerSpoke);
  float const               sampPhase = p.Tsamp * Δf * 2 * M_PI;
  float const               startPhase = p.samplesGap * sampPhase;
  float const               endPhase = (p.samplesGap + p.samplesPerSpoke - 1) * sampPhase;
  float const               sampDecay = p.Tsamp * R2;
  float const               startDecay = p.samplesGap * sampDecay;
  float const               endDecay = (p.samplesGap + p.samplesPerSpoke - 1) * sampDecay;
  em = Eigen::VectorXcf::LinSpaced(p.samplesPerSpoke, startPhase * 1if - startDecay, endPhase * 1if - endDecay).array().exp();
  // Log::Print("Δf {} startPhase {} endPhase {}", Δf, startPhase, endPhase);
  return e;
}

auto SegmentedZTE::E(float const R, float const t) const -> Eigen::Matrix2f
{
  float const     e = exp(-R * t);
  Eigen::Matrix2f Et;
  Et << e, 1 - e, 0.f, 1.f;
  return Et;
}

auto SegmentedZTE::A(float const a) const -> Eigen::Matrix2f
{
  Eigen::Matrix2f A1;
  A1 << std::cos(a), 0.f, 0.f, 1.f;
  return A1;
}

auto SegmentedZTE::Eseg(float const R1, float const B1) const -> Eigen::Matrix2f
{
  Eigen::Matrix2f const E1 = E(R1, p.TR), Eramp = E(R1, p.Tramp), Essi = E(R1, p.Tssi), A1 = A(p.alpha * B1);
  return (Essi * Eramp * (E1 * A1).pow(p.spokesPerSeg) * E1.pow(p.spokesSpoil) * Eramp);
}

auto SegmentedZTE::inv(float const Q) const -> Eigen::Matrix2f
{
  Eigen::Matrix2f inv;
  inv << -Q, 0.f, 0.f, 1.f;
  return inv;
}

void SegmentedZTE::segment(Index &tp, float &t, Re1 &ts, Eigen::Vector2f &Mz, Cx1 &s0, float const R1, float const B1) const
{
  auto const Eramp = E(R1, p.Tramp);
  auto const E1 = E(R1, p.TR);
  auto const Essi = E(R1, p.Tssi);
  auto const A1 = A(B1 * p.alpha);
  auto const sina = std::sin(B1 * p.alpha);

  if (preseg) {
    s0(tp) = Mz(0) * sina;
    ts(tp++) = t;
  }
  t += p.Tramp;
  Mz = Eramp * Mz;
  for (Index ii = 0; ii < p.spokesSpoil; ii++) {
    t += p.TR;
    Mz = E1 * Mz;
  }
  for (Index ii = 0; ii < p.spokesPerSeg; ii++) {
    s0(tp) = Mz(0) * sina;
    ts(tp++) = t;
    t += p.TR;
    Mz = E1 * A1 * Mz;
  }
  t += p.Tramp + p.Tssi;
  Mz = Essi * Eramp * Mz;
}

} // namespace rl