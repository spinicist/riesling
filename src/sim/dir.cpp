#include "dir.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

DIR::DIR(Settings const s)
  : Sequence{s}
{
  Log::Print(
    "DIR simulation TI1 {} Trec {} ⍺ {} TR {} SPG {}", settings.TI, settings.Trec, settings.alpha, settings.TR, settings.spg);
}

Index DIR::length() const { return settings.spg * settings.gps; }

Eigen::ArrayXXf DIR::parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const
{
  return Parameters::T1B1η(nsamp, lo, hi);
}

Eigen::ArrayXf DIR::simulate(Eigen::ArrayXf const &p) const
{
  float const T1 = p(0);
  float const B1 = p(1);
  float const η = p(2);
  Eigen::ArrayXf dynamic(settings.spg * settings.gps);

  Eigen::Matrix2f inv;
  inv << -η, 0.f, 0.f, 1.f;

  float const R1 = 1.f / T1;
  Eigen::Matrix2f E1, Einv, Einv2, Eramp, Essi, Erec, Esat;
  float const e1 = exp(-R1 * settings.TR);
  float const einv = exp(-R1 * settings.TI);
  float const eramp = exp(-R1 * settings.Tramp);
  float const essi = exp(-R1 * settings.Tssi);
  float const erec = exp(-R1 * settings.Trec);
  float const esat = exp(-R1 * settings.Tsat);
  E1 << e1, 1.f - e1, 0.f, 1.f;
  Einv << einv, 1.f - einv, 0.f, 1.f;
  Eramp << eramp, 1.f - eramp, 0.f, 1.f;
  Essi << essi, 1.f - essi, 0.f, 1.f;
  Erec << erec, 1.f - erec, 0.f, 1.f;
  Esat << esat, 1.f - esat, 0.f, 1.f;

  float const cosa = cos(B1 * settings.alpha * M_PI / 180.f);
  float const sina = sin(B1 * settings.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const grp = (Essi * Eramp * (E1 * A).pow(settings.spg) * Eramp * Esat);
  Eigen::Matrix2f const SS =
    Einv * inv * Erec * grp.pow(settings.gps - settings.gprep2) * Einv * inv * grp.pow(settings.gprep2);
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  for (Index ig = 0; ig < settings.gprep2; ig++) {
    Mz = Esat * Mz;
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  Mz = Einv * inv * Mz;
  for (Index ig = 0; ig < settings.gps - settings.gprep2; ig++) {
    Mz = Esat * Mz;
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != settings.spg * settings.gps) {
    Log::Fail("Programmer error");
  }
  return dynamic;
}

DIR2::DIR2(Settings const s)
  : Sequence{s}
{
  Log::Print(
    "DIR2 simulation TI1 {} Trec {} ⍺ {} TR {} SPG {} TE {}", settings.TI, settings.Trec, settings.alpha, settings.TR, settings.spg, settings.TE);
}

Index DIR2::length() const { return settings.spg * settings.gps; }

Eigen::ArrayXXf DIR2::parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const
{
  return Parameters::T1T2PD(nsamp, lo, hi);
}

Eigen::ArrayXf DIR2::simulate(Eigen::ArrayXf const &p) const
{
  float const T1 = p(0);
  float const T2 = p(1);
  float const PD = p(2);
  float const B1 = 0.7;
  float const η = 1.0f;
  Eigen::ArrayXf dynamic(settings.spg * settings.gps);

  Eigen::Matrix2f inv;
  inv << -η, 0.f, 0.f, 1.f;

  float const R1 = 1.f / T1;
  float const R2 = 1.f / T2;
  Eigen::Matrix2f E1, E2, Einv, Eramp, Essi, Erec, Esat;
  float const e1 = exp(-R1 * settings.TR);
  float const e2 = exp(-R2 * settings.TE);
  float const einv = exp(-R1 * settings.TI);
  float const eramp = exp(-R1 * settings.Tramp);
  float const essi = exp(-R1 * settings.Tssi);
  float const erec = exp(-R1 * settings.Trec);
  float const esat = exp(-R1 * settings.Tsat);
  E1 << e1, 1.f - e1, 0.f, 1.f;
  Einv << einv, 1.f - einv, 0.f, 1.f;
  E2 << -e2, 0.f, 0.f, 1.f;
  Eramp << eramp, 1.f - eramp, 0.f, 1.f;
  Essi << essi, 1.f - essi, 0.f, 1.f;
  Erec << erec, 1.f - erec, 0.f, 1.f;
  Esat << esat, 1.f - esat, 0.f, 1.f;

  float const cosa = cos(B1 * settings.alpha * M_PI / 180.f);
  float const sina = sin(B1 * settings.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const grp = (Essi * Eramp * (E1 * A).pow(settings.spg) * Eramp * Esat);
  Eigen::Matrix2f const SS =
    Einv * inv * Erec * grp.pow(settings.gps - settings.gprep2) * E2 * grp.pow(settings.gprep2);
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  Mz *= PD;
  for (Index ig = 0; ig < settings.gprep2; ig++) {
    Mz = Esat * Mz;
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  Mz = E2 * Mz;
  for (Index ig = 0; ig < settings.gps - settings.gprep2; ig++) {
    Mz = Esat * Mz;
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != settings.spg * settings.gps) {
    Log::Fail("Programmer error");
  }
  return dynamic;
}

} // namespace rl
