#include "ir.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

IR::IR(Settings const &s)
  : Sequence{s}
{
}

auto IR::length() const -> Index
{
  return settings.spg * settings.gps;
}

auto IR::parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const -> Eigen::ArrayXXf
{
  return Parameters::T1(nsamp, lo, hi);
}

auto IR::simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf
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


IR2::IR2(Settings const &s)
  : Sequence{s}
{
}

auto IR2::length() const -> Index
{
  return settings.spg * settings.gps;
}

auto IR2::parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const -> Eigen::ArrayXXf
{
  return Parameters::T1B1η(nsamp, lo, hi);
}

auto IR2::simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf
{
  float const T1 = p(0);
  float const B1 = p(1);
  float const η = p(2);
  Eigen::ArrayXf dynamic(settings.spg * settings.gps);

  Eigen::Matrix2f inv;
  inv << -η, 0.f, 0.f, 1.f;

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

  float const cosa1 = cos(B1 * settings.alpha * settings.ascale * M_PI / 180.f);
  float const sina1 = sin(B1 * settings.alpha * settings.ascale * M_PI / 180.f);
  float const cosa2 = cos(B1 * settings.alpha * M_PI / 180.f);
  float const sina2 = sin(B1 * settings.alpha * M_PI / 180.f);

  Eigen::Matrix2f A1, A2;
  A1 << cosa1, 0.f, 0.f, 1.f;
  A2 << cosa2, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const grp1 = (Essi * Eramp * (E1 * A1).pow(settings.spg) * Eramp);
  Eigen::Matrix2f const grp2 = (Essi * Eramp * (E1 * A2).pow(settings.spg) * Eramp);
  Eigen::Matrix2f const seg = (grp2 * grp1).pow(settings.gps / 2);
  Eigen::Matrix2f const SS = Einv * inv * Erec * seg;
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  for (Index ig = 0; ig < settings.gps/2; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina1;
      Mz = E1 * A1 * Mz;
    }
    Mz = Essi * Eramp * Mz;
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina2;
      Mz = E1 * A2 * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != settings.spg * settings.gps) {
    Log::Fail("Programmer error");
  }
  return dynamic;
}

} // namespace rl
