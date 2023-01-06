#include "mprage.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

MPRAGE::MPRAGE(Settings const &s)
  : Sequence{s}
{
}

Index MPRAGE::length() const
{
  return settings.spg * settings.gps;
}

Eigen::ArrayXXf MPRAGE::parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const
{
  return Parameters::T1(nsamp, lo, hi);
}

Eigen::ArrayXf MPRAGE::simulate(Eigen::ArrayXf const &p) const
{
  float const T1 = p(0);
  Eigen::ArrayXf dynamic(settings.spg * settings.gps);

  Eigen::Matrix2f inv;
  inv << -1.f, 0.f, 0.f, 1.f;

  float const R1 = 1.f / T1;
  Eigen::Matrix2f E1, Einv, Eramp, Essi, Erec;
  float const e1 = exp(-R1 * settings.TR);
  float const einv = exp(-R1 * settings.TI);
  float const eramp = exp(-R1 * settings.Tramp);
  float const essi = exp(-R1 * settings.Tssi);
  float const erec = exp(-R1 * settings.Trec);
  E1 << e1, 1 - e1, 0.f, 1.f;
  Einv << einv, 1 - einv, 0.f, 1.f;
  Eramp << eramp, 1 - eramp, 0.f, 1.f;
  Essi << essi, 1 - essi, 0.f, 1.f;
  Erec << erec, 1 - erec, 0.f, 1.f;

  float const cosa = cos(settings.alpha * M_PI / 180.f);
  float const sina = sin(settings.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const seg = (Essi * Eramp * (E1 * A).pow(settings.spg) * Eramp).pow(settings.gps);
  Eigen::Matrix2f const SS = Einv * inv * Erec * seg;
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  for (Index ig = 0; ig < settings.gps; ig++) {
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
