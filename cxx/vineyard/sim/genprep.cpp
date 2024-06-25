#include "genprep.hpp"

#include "parameter.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

GenPrep::GenPrep(Settings const &s)
  : Sequence{s}
{
}

auto GenPrep::length() const -> Index { return (settings.spokesPerSeg + settings.k0) * settings.segsPerPrep; }

auto GenPrep::simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf
{
  float const    T1 = p(0);
  float const    β = p(1);
  Eigen::ArrayXf dynamic(length());

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

  // Get steady state after prep-pulse for fGenPrepst segment
  Eigen::Matrix2f const seg =
    (Essi * (E1 * A).pow(settings.k0) * Eramp * (E1 * A).pow(settings.spokesPerSeg + settings.spokesSpoil) * Eramp)
      .pow(settings.segsPerPrep);
  Eigen::Matrix2f const SS = Eprep * prep * Erec * seg;
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

GenPrep2::GenPrep2(Settings const &s)
  : Sequence{s}
{
}

auto GenPrep2::length() const -> Index { return settings.spokesPerSeg * settings.segsPerPrepKeep; }

auto GenPrep2::simulate(Eigen::ArrayXf const &p) const -> Eigen::ArrayXf
{
  float const    R1 = 1.f / p(0);
  float const    β1 = p(1);
  float const    β2 = p(2);

  Eigen::Matrix2f prep1, prep2;
  prep1 << β1, 0.f, 0.f, 1.f;
  prep2 << β2, 0.f, 0.f, 1.f;
 
  Eigen::ArrayXf dynamic(settings.spokesPerSeg * settings.segsPerPrepKeep);

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
  for (Index ig = 0; ig < settings.segsPrep2; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spokesSpoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < settings.spokesPerSeg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  Mz = Essi * Erec * prep2 * Mz;
  for (Index ig = 0; ig < (settings.segsPerPrepKeep - settings.segsPrep2); ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spokesSpoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < settings.spokesPerSeg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != settings.spokesPerSeg * settings.segsPerPrepKeep) { Log::Fail("Programmer error"); }
  return dynamic;
}


} // namespace rl
