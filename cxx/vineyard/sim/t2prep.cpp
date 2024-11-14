#include "t2prep.hpp"

#include "log.hpp"
#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

T2Prep::T2Prep(Pars const &s, bool const pt)
  : SegmentedZTE(s, pt)
{
}

auto T2Prep::nTissueParameters() const -> Index { return 4; }

auto T2Prep::simulate(Eigen::ArrayXf const &pars) const -> Cx2
{
  if (pars.size() != nTissueParameters()) {
    throw Log::Failure("Sim", "Must have {} parameters T1 T2 Δf B1", nTissueParameters());
  }
  float const R1 = 1.f / pars(0);
  float const R2 = 1.f / pars(1);
  float const Δf = pars(2);
  float const B1 = pars(3);

  Eigen::Matrix2f const E2 = ET2p(R2, p.TE), R = E(R1, p.Trec), seg = Eseg(R1, B1);
  Eigen::Matrix2f const SS = E2 * R * seg.pow(p.segsPerPrep);
  float const           m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Eigen::Vector2f Mz{m_ss, 1.f};
  Cx1             s0(traces());
  Index           tp = 0;
  for (Index ig = 0; ig < p.segsPerPrep; ig++) {
    segment(tp, Mz, s0, R1, B1);
  }
  if (tp != traces()) { throw Log::Failure("Sim", "Programmer error"); }
  return readout(pars(1), Δf).contract(s0, Eigen::array<Eigen::IndexPair<Index>, 0>());
}

auto T2Prep::timepoints() const -> Re1
{
  Re1   ts(traces());
  Index tp = 0;
  float t = 0.f;
  for (Index ig = 0; ig < p.segsPerPrep; ig++) {
    segmentTimepoints(tp, t, ts);
  }
  if (tp != traces()) { throw Log::Failure("Sim", "Programmer error"); }
  return ts;
}

} // namespace rl
