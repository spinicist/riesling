#include "Prep.hpp"

#include "parameter.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

Prep::Prep(Settings const &s)
  : Sequence{s}
{
}

auto Prep::length() const -> Index { return (settings.spokesPerSeg + settings.k0) * settings.segsPerPrep; }

auto Prep::simulate(Eigen::ArrayXf const &p) const -> Cx2
{
  if (p.size() != 3) { Log::Fail("Must have 3 parameters T1 β Δf"); }
  float const T1 = p(0);
  float const β = p(1);
  float const Δf = p(2);

  Eigen::Matrix2f prep;
  prep << β, 0.f, 0.f, 1.f;

  float const     R1 = 1.f / T1;
  Eigen::Matrix2f E1, Eprep, Eramp, Essi, Erec;
  float const     e1 = exp(-R1 * settings.TR);
  float const     eprep = exp(-R1 * settings.TI);
  float const     eramp = exp(-R1 * settings.Tramp);
  float const     essi = exp(-R1 * settings.Tssi);
  float const     erec = exp(-R1 * settings.Trec);
  E1 << e1, 1 - e1, 0.f, 1.f;
  Eprep << eprep, 1 - eprep, 0.f, 1.f;
  Eramp << eramp, 1 - eramp, 0.f, 1.f;
  Essi << essi, 1 - essi, 0.f, 1.f;
  Erec << erec, 1 - erec, 0.f, 1.f;

  float const cosa = cos(settings.alpha * M_PI / 180.f);
  float const sina = sin(settings.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for fPrepst segment
  Eigen::Matrix2f const seg =
    (Essi * (E1 * A).pow(settings.k0) * Eramp * (E1 * A).pow(settings.spokesPerSeg + settings.spokesSpoil) * Eramp)
      .pow(settings.segsPerPrep);
  Eigen::Matrix2f const SS = Eprep * prep * Erec * seg;
  float const           m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index           tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  Cx1             s0(length());
  for (Index ig = 0; ig < settings.segsPerPrep; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spokesSpoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < settings.spokesPerSeg; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
    for (Index ii = 0; ii < settings.k0; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
  }
  if (tp != length()) { Log::Fail("Programmer error"); }
  return offres(Δf).contract(s0, Eigen::array<Eigen::IndexPair<Index>, 0>());
}

Prep2::Prep2(Settings const &s)
  : Sequence{s}
{
}

auto Prep2::length() const -> Index { return settings.spokesPerSeg * settings.segsKeep; }

auto Prep2::simulate(Eigen::ArrayXf const &p) const -> Cx2
{
  if (p.size() != 4) { Log::Fail("Must have 4 parameters T1 β1 β2 Δf"); }
  float const R1 = 1.f / p(0);
  float const β1 = p(1);
  float const β2 = p(2);
  float const Δf = p(3);

  Eigen::Matrix2f prep1, prep2;
  prep1 << β1, 0.f, 0.f, 1.f;
  prep2 << β2, 0.f, 0.f, 1.f;

  Eigen::Matrix2f E1, Eramp, Essi, Er, Erec;
  float const     e1 = exp(-R1 * settings.TR);
  float const     eramp = exp(-R1 * settings.Tramp);
  float const     essi = exp(-R1 * settings.Tssi);
  float const     erec = exp(-R1 * settings.Trec);
  E1 << e1, 1 - e1, 0.f, 1.f;
  Eramp << eramp, 1 - eramp, 0.f, 1.f;
  Essi << essi, 1 - essi, 0.f, 1.f;
  Erec << erec, 1 - erec, 0.f, 1.f;

  float const cosa = cos(settings.alpha * M_PI / 180.f);
  float const sina = sin(settings.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state before first read-out
  Eigen::Matrix2f const grp = (Essi * Eramp * (E1 * A).pow(settings.spokesPerSeg + settings.spokesSpoil) * Eramp);
  Eigen::Matrix2f const SS =
    Essi * prep1 * grp.pow(settings.segsPerPrep - settings.segsPrep2) * Essi * prep2 * grp.pow(settings.segsPrep2);
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index           tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  Cx1             s0(length());
  for (Index ig = 0; ig < settings.segsPrep2; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spokesSpoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < settings.spokesPerSeg; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  Mz = Essi * Erec * prep2 * Mz;
  for (Index ig = 0; ig < (settings.segsKeep - settings.segsPrep2); ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spokesSpoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < settings.spokesPerSeg; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != settings.spokesPerSeg * settings.segsKeep) { Log::Fail("Programmer error"); }
  return offres(Δf).contract(s0, Eigen::array<Eigen::IndexPair<Index>, 0>());
}

} // namespace rl
