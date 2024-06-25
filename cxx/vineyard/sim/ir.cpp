#include "ir.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

IR::IR(Settings const &s)
  : Sequence{s}
{
}

auto IR::length() const -> Index { return (settings.spokesPerSeg + settings.k0) * settings.segsPerPrep; }

auto IR::simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf
{
  float const    T1 = p(0);
  Eigen::ArrayXf dynamic(length());

  Eigen::Matrix2f inv;
  inv << -1.f, 0.f, 0.f, 1.f;

  float const     R1 = 1.f / T1;
  Eigen::Matrix2f E1, Einv, Eramp, Essi, Erec;
  float const     e1 = exp(-R1 * settings.TR);
  float const     einv = exp(-R1 * settings.TI);
  float const     eramp = exp(-R1 * settings.Tramp);
  float const     essi = exp(-R1 * settings.Tssi);
  float const     erec = exp(-R1 * settings.Trec);
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
  Eigen::Matrix2f const seg =
    (Essi * (E1 * A).pow(settings.k0) * Eramp * (E1 * A).pow(settings.spokesPerSeg + settings.spokesSpoil) * Eramp)
      .pow(settings.segsPerPrep);
  Eigen::Matrix2f const SS = Einv * inv * Erec * seg;
  float const           m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index           tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  for (Index ig = 0; ig < settings.segsPerPrep; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spokesSpoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < settings.spokesPerSeg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
    for (Index ii = 0; ii < settings.k0; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
  }
  if (tp != length()) { Log::Fail("Programmer error"); }
  return dynamic;
}

IR2::IR2(Settings const &s)
  : Sequence{s}
{
  if (settings.segsPerPrep < 2) { Log::Fail("Surely you had 2 segments per prep"); }
  if (settings.segsPerPrep % 2 == 1) { Log::Fail("Surely you had an even number of segments per prep"); }
}

auto IR2::length() const -> Index { return (settings.spokesPerSeg + settings.k0) * settings.segsPerPrep; }

auto IR2::simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf
{
  float const    T1 = p(0);
  float const    B1 = p(1);
  Eigen::ArrayXf dynamic(length());

  Eigen::Matrix2f inv;
  inv << cos(M_PI * B1), 0.f, 0.f, 1.f;

  float const     R1 = 1.f / T1;
  Eigen::Matrix2f E1, Einv, Eramp, Essi, Erec;
  float const     e1 = exp(-R1 * settings.TR);
  float const     einv = exp(-R1 * settings.TI);
  float const     eramp = exp(-R1 * settings.Tramp);
  float const     essi = exp(-R1 * settings.Tssi);
  float const     erec = exp(-R1 * settings.Trec);
  E1 << e1, 1 - e1, 0.f, 1.f;
  Einv << einv, 1 - einv, 0.f, 1.f;
  Eramp << eramp, 1 - eramp, 0.f, 1.f;
  Essi << essi, 1 - essi, 0.f, 1.f;
  Erec << erec, 1 - erec, 0.f, 1.f;

  float const a = B1 * settings.alpha * M_PI / 180.f;
  float const cosa = cos(a);
  float const sina = sin(a);
  float const a2 = a / settings.ascale;
  float const cosa2 = cos(a2);
  float const sina2 = sin(a2);

  Eigen::Matrix2f A, A2;
  A << cosa, 0.f, 0.f, 1.f;
  A2 << cosa2, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const seg =
    (Essi * (E1 * A).pow(settings.k0) * Eramp * (E1 * A).pow(settings.spokesPerSeg + settings.spokesSpoil) * Eramp);
  Eigen::Matrix2f const seg2 =
    (Essi * (E1 * A2).pow(settings.k0) * Eramp * (E1 * A2).pow(settings.spokesPerSeg + settings.spokesSpoil) * Eramp);
  Eigen::Matrix2f const SS = Einv * inv * Erec * (seg2 * seg).pow(settings.segsPerPrep / 2);
  float const           m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index           tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  for (Index ig = 0; ig < settings.segsPerPrep / 2; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spokesSpoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < settings.spokesPerSeg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
    for (Index ii = 0; ii < settings.k0; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Eramp * Essi * Eramp * Mz;
    for (Index ii = 0; ii < settings.spokesSpoil; ii++) {
      Mz = E1 * A2 * Mz;
    }
    for (Index ii = 0; ii < settings.spokesPerSeg; ii++) {
      dynamic(tp++) = Mz(0) * sina2;
      Mz = E1 * A2 * Mz;
    }
    Mz = Essi * Eramp * Mz;
    for (Index ii = 0; ii < settings.k0; ii++) {
      dynamic(tp++) = Mz(0) * sina2;
      Mz = E1 * A2 * Mz;
    }
  }
  if (tp != length()) { Log::Fail("Programmer error"); }
  return dynamic;
}

} // namespace rl
