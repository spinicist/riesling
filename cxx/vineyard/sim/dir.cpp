#include "dir.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

DIR::DIR(Parameters const s)
  : SegmentedZTE{s}
{
  Log::Print("Sim", "DIR SegmentedZTE");
}

Index DIR::traces() const { return p.spokesPerSeg * p.segsKeep; }

auto DIR::simulate(Eigen::ArrayXf const &p) const -> Cx2
{
  if (p.size() != 4) { throw Log::Failure("Sim", "Need 4 parameters T1 T2 Δf"); }
  float const T1 = p(0);
  float const T2 = p(1);
  float const Δf = p(2);
  float const Q = p(3);

  Eigen::Matrix2f inv;
  inv << -1, 0.f, 0.f, 1.f;

  float const     R1 = 1.f / T1;
  float const     R2 = 1.f / T2;
  Eigen::Matrix2f E1, E2, Einv, Eramp, Essi, Erec;
  float const     e1 = exp(-R1 * p.TR);
  float const     e2 = exp(-R2 * p.TE);
  float const     einv = exp(-R1 * p.TI);
  float const     eramp = exp(-R1 * p.Tramp);
  float const     essi = exp(-R1 * p.Tssi);
  float const     erec = exp(-R1 * p.Trec);
  E1 << e1, 1.f - e1, 0.f, 1.f;
  Einv << einv, 1.f - einv, 0.f, 1.f;
  E2 << -Q * e2, 0.f, 0.f, 1.f;
  Eramp << eramp, 1.f - eramp, 0.f, 1.f;
  Essi << essi, 1.f - essi, 0.f, 1.f;
  Erec << erec, 1.f - erec, 0.f, 1.f;

  float const cosa = cos(p.alpha * M_PI / 180.f);
  float const sina = sin(p.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const grp = (Essi * Eramp * (E1 * A).pow(p.spokesPerSeg) * E1.pow(p.spokesSpoil) * Eramp);
  Eigen::Matrix2f const SS =
    Einv * inv * Erec * grp.pow(p.segsPerPrep - p.segsPrep2) * E2 * grp.pow(p.segsPrep2);
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index           tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  Cx1             s0(traces());
  for (Index ig = 0; ig < p.segsPrep2; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < p.spokesSpoil; ii++) {
      Mz = E1 * Mz;
    }
    for (Index ii = 0; ii < p.spokesPerSeg; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  Mz = E2 * Mz;
  for (Index ig = 0; ig < p.segsKeep - p.segsPrep2; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < p.spokesSpoil; ii++) {
      Mz = E1 * Mz;
    }
    for (Index ii = 0; ii < p.spokesPerSeg; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != p.spokesPerSeg * p.segsKeep) { throw Log::Failure("Sim", "Programmer error"); }
  return offres(Δf).contract(s0, Eigen::array<Eigen::IndexPair<Index>, 0>());
}

} // namespace rl
