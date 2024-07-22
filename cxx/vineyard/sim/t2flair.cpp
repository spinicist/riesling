#include "t2flair.hpp"

#include "log.hpp"
#include "parameter.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

T2FLAIR::T2FLAIR(Settings const &s)
  : Sequence{s}
{
}

auto T2FLAIR::length() const -> Index { return settings.spokesPerSeg * settings.segsKeep; }

auto T2FLAIR::simulate(Eigen::ArrayXf const &p) const -> Cx2
{
  if (p.size() != 4) { Log::Fail("Need 3 parameters T1 T2 Δf"); }
  float const R1 = 1.f / p(0);
  float const R2 = 1.f / p(1);
  float const Δf = p(2);
  float const Q = p(3);

  Eigen::Matrix2f inv;
  inv << -Q, 0.f, 0.f, 1.f;

  Eigen::Matrix2f E1, E2, Eramp, Essi, Er, Erec;
  float const     e1 = exp(-R1 * settings.TR);
  float const     eramp = exp(-R1 * settings.Tramp);
  float const     essi = exp(-R1 * settings.Tssi);
  float const     erec = exp(-R1 * settings.Trec);
  float const     e2 = exp(-R2 * settings.TE);
  E1 << e1, 1 - e1, 0.f, 1.f;
  E2 << e2, 0.f, 0.f, 1.f;
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
    Essi * E2 * inv * grp.pow(settings.segsPerPrep - settings.segsPrep2) * Essi * E2 * grp.pow(settings.segsPrep2);
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index           tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  Cx1             s0(settings.spokesPerSeg * settings.segsKeep);
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
  Mz = Essi * Erec * E2 * Mz;
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
