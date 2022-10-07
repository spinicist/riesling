#include "t1t2.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

T1T2Prep::T1T2Prep(Settings const &s)
  : Sequence{s}
{
}

Index T1T2Prep::length() const
{
  return 2 * settings.spg * settings.gps;
}

Eigen::ArrayXXf T1T2Prep::parameters(Index const nS) const
{
  float const R1lo = 1.f / 0.25f;
  float const R1hi = 1.f / 5.0f;
  float const R2lo = 1.f / 0.02f;
  float const R2hi = 1.f;
  Index const nT = std::floor(std::pow(nS, 0.33f));
  auto const R1s = Eigen::ArrayXf::LinSpaced(nT, R1lo, R1hi);
  auto const R2s = Eigen::ArrayXf::LinSpaced(nT, R2lo, R2hi);
  Index nAct = 0;
  Eigen::ArrayXXf p(2, nS);
  for (Index i1 = 0; i1 < nT; i1++) {
    for (Index i2 = 0; i2 < nT; i2++) {
      if (R2s(i2) > R1s(i1)) {
        p(0, nAct) = 1.f / R1s(i1);
        p(1, nAct) = 1.f / R2s(i2);
        nAct++;
      }
    }
  }
  p.conservativeResize(2, nAct);
  Eigen::ArrayXXf p2(3, nAct * nT);
  auto const B1s = Eigen::ArrayXf::LinSpaced(nT, 0.5f, 1.5f);
  Index ii = 0;
  for (Index ib = 0; ib < nT; ib++) {
    for (Index it = 0; it < nAct; it++) {
      p(0, ii) = p(0, it);
      p(1, ii) = p(1, it);
      p(2, ii) = B1s(ib);
      ii++;
    }
  }
  return p2;
}

Eigen::ArrayXf T1T2Prep::simulate(Eigen::ArrayXf const &p) const
{
  float const T1 = p(0);
  float const T2 = p(1);
  float const B1v = p(2);

  Eigen::ArrayXf dynamic(settings.spg * settings.gps * 2);

  // Set up matrices
  Eigen::Matrix2f inv;
  inv << -1.f, 0.f, 0.f, 1.f;
  float const R1 = 1.f / T1;
  Eigen::Matrix2f E1, Eramp, Essi;
  float const e1 = exp(-R1 * settings.TR);
  float const eramp = exp(-R1 * settings.Tramp);
  float const essi = exp(-R1 * settings.Tssi);
  E1 << e1, 1 - e1, 0.f, 1.f;
  Eramp << eramp, 1 - eramp, 0.f, 1.f;
  Essi << essi, 1 - essi, 0.f, 1.f;

  float const cosa1 = cos(B1v * settings.alpha * M_PI / 180.f);
  float const sina1 = sin(B1v * settings.alpha * M_PI / 180.f);
  float const cosa2 = cos(B1v * settings.alpha * settings.ascale * M_PI / 180.f);
  float const sina2 = sin(B1v * settings.alpha * settings.ascale * M_PI / 180.f);

  Eigen::Matrix2f A1, A2;
  A1 << cosa1, 0.f, 0.f, 1.f;
  A2 << cosa2, 0.f, 0.f, 1.f;

  float const R2 = 1.f / T2;
  Eigen::Matrix2f E2;
  E2 << exp(-R2 * settings.TE), 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const grp1 = (Essi * Eramp * (E1 * A1).pow(settings.spg) * Eramp);
  Eigen::Matrix2f const grp2 = (Essi * Eramp * (E1 * A2).pow(settings.spg) * Eramp);
  Eigen::Matrix2f const seg = (grp2 * grp1).pow(settings.gps / 2);
  Eigen::Matrix2f const SS = Essi * inv * E2 * seg * Essi * E2 * seg;
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  for (Index ig = 0; ig < (settings.gps / 2); ig++) {
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
  Mz = E2 * Mz;
  for (Index ig = 0; ig < (settings.gps / 2); ig++) {
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
  if (tp != (settings.spg * settings.gps * 2)) {
    Log::Fail("Programmer error");
  }
  return dynamic;
}

} // namespace rl
