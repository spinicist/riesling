#include "ir.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

IR::IR(Parameters const &s)
  : SegmentedZTE{s}
{
}

auto IR::traces() const -> Index { return (p.spokesPerSeg + p.k0) * p.segsPerPrep; }

auto IR::simulate(Eigen::ArrayXf const &p) const -> Cx2
{
  if (p.rows() != 3) { throw Log::Failure("Sim", "Parameters must be T1 Δf Q"); }
  float const T1 = p(0);
  float const Δf = p(1);
  float const Q = p(2);

  Eigen::Matrix2f inv;
  inv << -Q, 0.f, 0.f, 1.f;

  float const     R1 = 1.f / T1;
  Eigen::Matrix2f E1, Einv, Eramp, Essi, Erec;
  float const     e1 = exp(-R1 * p.TR);
  float const     einv = exp(-R1 * p.TI);
  float const     eramp = exp(-R1 * p.Tramp);
  float const     essi = exp(-R1 * p.Tssi);
  float const     erec = exp(-R1 * p.Trec);
  E1 << e1, 1 - e1, 0.f, 1.f;
  Einv << einv, 1 - einv, 0.f, 1.f;
  Eramp << eramp, 1 - eramp, 0.f, 1.f;
  Essi << essi, 1 - essi, 0.f, 1.f;
  Erec << erec, 1 - erec, 0.f, 1.f;

  float const cosa = cos(p.alpha * M_PI / 180.f);
  float const sina = sin(p.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const seg =
    (Essi * (E1 * A).pow(p.k0) * Eramp * (E1 * A).pow(p.spokesPerSeg + p.spokesSpoil) * Eramp)
      .pow(p.segsPerPrep);
  Eigen::Matrix2f const SS = Einv * inv * Erec * seg;
  float const           m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index           tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  Cx1             s0(traces());
  for (Index ig = 0; ig < p.segsPerPrep; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < p.spokesSpoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < p.spokesPerSeg; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
    for (Index ii = 0; ii < p.k0; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
  }
  if (tp != traces()) { throw Log::Failure("Sim", "Programmer error"); }
  return offres(Δf).contract(s0, Eigen::array<Eigen::IndexPair<Index>, 0>());
}

IR2::IR2(Parameters const &s)
  : SegmentedZTE{s}
{
}

auto IR2::traces() const -> Index { return (p.spokesPerSeg + p.k0) * p.segsPerPrep; }

auto IR2::simulate(Eigen::ArrayXf const &p) const -> Cx2
{
  if (p.rows() != 4) { throw Log::Failure("Sim", "Parameters must be T1 T2 Δf Q"); }
  float const T1 = p(0);
  float const T2 = p(1);
  float const Δf = p(2);
  float const Q = p(3);

  Eigen::Matrix2f inv;
  inv << -Q, 0.f, 0.f, 1.f;

  float const     R1 = 1.f / T1;
  Eigen::Matrix2f E1, Einv, Eramp, Essi, Erec;
  float const     e1 = exp(-R1 * p.TR);
  float const     einv = exp(-R1 * p.TI);
  float const     eramp = exp(-R1 * p.Tramp);
  float const     essi = exp(-R1 * p.Tssi);
  float const     erec = exp(-R1 * p.Trec);
  E1 << e1, 1 - e1, 0.f, 1.f;
  Einv << einv, 1 - einv, 0.f, 1.f;
  Eramp << eramp, 1 - eramp, 0.f, 1.f;
  Essi << essi, 1 - essi, 0.f, 1.f;
  Erec << erec, 1 - erec, 0.f, 1.f;

  float const cosa = cos(p.alpha * M_PI / 180.f);
  float const sina = sin(p.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const seg =
    (Essi * (E1 * A).pow(p.k0) * Eramp * (E1 * A).pow(p.spokesPerSeg) * E1.pow(p.spokesSpoil) * Eramp)
      .pow(p.segsPerPrep);
  Eigen::Matrix2f const SS = Einv * inv * Erec * seg;
  float const           m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index           tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  Cx1             s0(traces());
  for (Index ig = 0; ig < p.segsPerPrep; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < p.spokesSpoil; ii++) {
      Mz = E1 * Mz;
    }
    for (Index ii = 0; ii < p.spokesPerSeg; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
    for (Index ii = 0; ii < p.k0; ii++) {
      s0(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
  }
  if (tp != traces()) { throw Log::Failure("Sim", "Programmer error"); }
  return readout(T2, Δf).contract(s0, Eigen::array<Eigen::IndexPair<Index>, 0>());
}

} // namespace rl
