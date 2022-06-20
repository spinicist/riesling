#include "t1t2.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

Index T1T2Prep::length() const
{
  return 2 * seq.sps;
}

Eigen::ArrayXXf T1T2Prep::parameters(Index const nsamp) const
{
  Tissues tissues({Tissue{{T1wm, T2wm, B1}}, Tissue{{T1gm, T2gm, B1}}, Tissue{{T1csf, T2csf, B1}}});
  return tissues.values(nsamp);
}

Eigen::ArrayXf T1T2Prep::simulate(Eigen::ArrayXf const &p) const
{
  float const T1 = p(0);
  float const T2 = p(1);
  Index const spg = seq.sps / seq.gps; // Spokes per group

  Eigen::ArrayXf dynamic(seq.sps * 2);

  // Set up matrices
  Eigen::Matrix2f inv;
  inv << -1.f, 0.f, 0.f, 1.f;
  float const R1 = 1.f / T1;
  Eigen::Matrix2f E1, Eramp, Essi;
  float const e1 = exp(-R1 * seq.TR);
  float const eramp = exp(-R1 * seq.Tramp);
  float const essi = exp(-R1 * seq.Tssi);
  E1 << e1, 1 - e1, 0.f, 1.f;
  Eramp << eramp, 1 - eramp, 0.f, 1.f;
  Essi << essi, 1 - essi, 0.f, 1.f;

  float const cosa1 = cos(seq.alpha * M_PI / 180.f);
  float const sina1 = sin(seq.alpha * M_PI / 180.f);
  float const cosa2 = cos(seq.alpha * seq.ascale * M_PI / 180.f);
  float const sina2 = sin(seq.alpha * seq.ascale * M_PI / 180.f);

  Eigen::Matrix2f A1, A2;
  A1 << cosa1, 0.f, 0.f, 1.f;
  A2 << cosa2, 0.f, 0.f, 1.f;

  float const R2 = 1.f / T2;
  Eigen::Matrix2f E2;
  E2 << exp(-R2 * seq.TE), 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const grp1 = (Essi * Eramp * (E1 * A1).pow(spg) * Eramp);
  Eigen::Matrix2f const grp2 = (Essi * Eramp * (E1 * A2).pow(spg) * Eramp);
  Eigen::Matrix2f const seg = (grp2 * grp1).pow(seq.gps / 2);
  Eigen::Matrix2f const SS = Essi * inv * E2 * seg * Essi * E2 * seg;
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  for (Index ig = 0; ig < (seq.gps / 2); ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < spg; ii++) {
      dynamic(tp++) = Mz(0) * sina1;
      Mz = E1 * A1 * Mz;
    }
    Mz = Essi * Eramp * Mz;

    Mz = Eramp * Mz;
    for (Index ii = 0; ii < spg; ii++) {
      dynamic(tp++) = Mz(0) * sina2;
      Mz = E1 * A2 * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  Mz = E2 * Mz;
  for (Index ig = 0; ig < (seq.gps / 2); ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < spg; ii++) {
      dynamic(tp++) = Mz(0) * sina1;
      Mz = E1 * A1 * Mz;
    }
    Mz = Essi * Eramp * Mz;

    Mz = Eramp * Mz;
    for (Index ii = 0; ii < spg; ii++) {
      dynamic(tp++) = Mz(0) * sina2;
      Mz = E1 * A2 * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != (seq.sps * 2)) {
    Log::Fail("Programmer error");
  }
  return dynamic;
}

} // namespace rl
