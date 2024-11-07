#include "t2flair.hpp"

#include "log.hpp"
#include "parameter.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

T2FLAIR::T2FLAIR(Parameters const &s, bool const pt)
  : SegmentedZTE{s, pt}
{
}

auto T2FLAIR::simulate(Eigen::ArrayXf const &pars) const -> Sim
{
  if (pars.size() != 4) { throw Log::Failure("Sim", "Need 3 parameters T1 T2 Δf"); }
  float const R1 = 1.f / pars(0);
  float const R2 = 1.f / pars(1);
  float const Δf = pars(2);
  float const Q = pars(3);

  Eigen::Matrix2f const E2 = E(R2, p.TE), V = inv(Q);

  // Get steady state before first read-out
  Eigen::Matrix2f const seg = Eseg(R1, 1.f);
  Eigen::Matrix2f const SS = E2 * V * seg.pow(p.segsPerPrep - p.segsPrep2) * E2 * seg.pow(p.segsPrep2);
  float const           m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Eigen::Vector2f Mz{m_ss, 1.f};
  Cx1             s0(traces());
  Re1             ts(traces());
  Index           tp = 0;
  float           t = 0.f;
  for (Index ig = 0; ig < p.segsPrep2; ig++) {
    segment(tp, t, ts, Mz, s0, R1, 1.f);
  }
  t += p.Tssi + p.TE;
  Mz = E2 * Mz;
  for (Index ig = 0; ig < p.segsPrep2; ig++) {
    segment(tp, t, ts, Mz, s0, R1, 1.f);
  }
  if (tp != traces()) { throw Log::Failure("Sim", "Programmer error"); }
  return {readout(pars(1), Δf).contract(s0, Eigen::array<Eigen::IndexPair<Index>, 0>()), ts};
}

} // namespace rl
