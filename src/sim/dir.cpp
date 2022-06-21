#include "dir.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

DIR::DIR(Settings const s)
  : seq{s}
{
  Log::Print(
    "DIR simulation TI1 {} TI2 {} Trec {} ⍺ {} TR {} SPS {}", seq.TI, seq.TI2, seq.Trec, seq.alpha, seq.TR, seq.sps);
}

Index DIR::length() const
{
  return seq.sps;
}

Eigen::ArrayXXf DIR::parameters(Index const nsamp) const
{
  Tissues tissues({Tissue{{T1wm}}, Tissue{{T1gm}}, Tissue{{T1csf}}});
  return tissues.values(nsamp);
}

Eigen::ArrayXf DIR::simulate(Eigen::ArrayXf const &p) const
{
  float const T1 = p(0);
  Index const spg = seq.sps / seq.gps; // Spokes per group
  Eigen::ArrayXf dynamic(seq.sps);

  Eigen::Matrix2f inv;
  inv << -1.f, 0.f, 0.f, 1.f;

  float const R1 = 1.f / T1;
  Eigen::Matrix2f E1, Einv, Einv2, Eramp, Essi, Erec;
  float const e1 = exp(-R1 * seq.TR);
  float const einv = exp(-R1 * seq.TI);
  float const einv2 = exp(-R1 * seq.TI2);
  float const eramp = exp(-R1 * seq.Tramp);
  float const essi = exp(-R1 * seq.Tssi);
  float const erec = exp(-R1 * seq.Trec);
  E1 << e1, 1 - e1, 0.f, 1.f;
  Einv << einv, 1 - einv, 0.f, 1.f;
  Einv2 << einv2, 1 - einv2, 0.f, 1.f;
  Eramp << eramp, 1 - eramp, 0.f, 1.f;
  Essi << essi, 1 - essi, 0.f, 1.f;
  Erec << erec, 1 - erec, 0.f, 1.f;

  float const cosa = cos(seq.alpha * M_PI / 180.f);
  float const sina = sin(seq.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const seg = (Essi * Eramp * (E1 * A).pow(spg) * Eramp).pow(seq.gps);
  Eigen::Matrix2f const SS = Einv2 * inv * Einv * inv * Erec * seg;
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  for (Index ig = 0; ig < seq.gps; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < spg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != seq.sps) {
    Log::Fail("Programmer error");
  }
  return dynamic;
}

} // namespace rl
