#include "dir.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

DIR::DIR(Settings const s)
  : Sequence{s}
{
  Log::Print(
    "DIR simulation TI1 {} Trec {} ⍺ {} TR {} SPG {}", settings.TI, settings.Trec, settings.alpha, settings.TR, settings.spg);
}

Index DIR::length() const
{
  return settings.spg * settings.gps;
}

Eigen::ArrayXXf DIR::parameters(Index const nsamp) const
{
  return Parameters::T1η(nsamp);
}

Eigen::ArrayXf DIR::simulate(Eigen::ArrayXf const &p) const
{
  float const T1 = p(0);
  float const η = p(1);
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

  float const cosa = cos(settings.alpha * M_PI / 180.f);
  float const sina = sin(settings.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const grp = (Essi * Eramp * (E1 * A).pow(settings.spg) * Eramp * Esat);
  Eigen::Matrix2f const SS = Einv * inv * Erec * grp.pow(settings.gps - settings.gprep2) * Einv * inv * grp.pow(settings.gprep2);
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

} // namespace rl
