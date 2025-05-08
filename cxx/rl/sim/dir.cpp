#include "dir.hpp"

#include "../log/log.hpp"
#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

DIR::DIR(Pars const s, bool const pt)
  : SegmentedZTE(s, pt)
{
  Log::Print("Sim", "DIR SegmentedZTE");
}

auto DIR::nTissueParameters() const -> Index { return 5; }

auto DIR::simulate(Eigen::ArrayXf const &pars) const -> Cx2
{
  if (pars.size() != nTissueParameters()) { throw Log::Failure("Sim", "Need {} parameters T1 T2 Δf Q", nTissueParameters()); }
  float const R1 = 1.f / pars(0);
  float const R2 = 1.f / pars(1);
  float const Δf = pars(2);
  float const Q = pars(3);
  float const B1 = pars(4);

  Eigen::Matrix2f const E2 = ET2p(R2, p.TE, true, Q), Ei = E(R1, p.TI), Er = E(R1, p.Trec), V1 = inv(Q), seg = Eseg(R1, B1);
  Eigen::Matrix2f const SS = Ei * V1 * Er * seg.pow(p.segsPerPrep - p.segsPrep2) * E2 * seg.pow(p.segsPrep2);
  float const           m_ss = SS(0, 1) / (1.f - SS(0, 0));
  Eigen::Vector2f       Mz{m_ss, 1.f};
  Cx1                   s0(traces());
  Index                 tp = 0;
  for (Index ig = 0; ig < p.segsPrep2; ig++) {
    segment(tp, Mz, s0, R1, B1);
  }
  Mz = E2 * Mz;
  for (Index ig = 0; ig < (p.segsPerPrep - p.segsPrep2); ig++) {
    segment(tp, Mz, s0, R1, B1);
  }
  if (tp != traces()) { throw Log::Failure("Sim", "Programmer error"); }
  return readout(pars(1), Δf).contract(s0, Eigen::array<Eigen::IndexPair<Index>, 0>());
}

auto DIR::timepoints() const -> Re1
{
  Re1   ts(traces());
  Index tp = 0;
  float t = 0.f;
  for (Index ig = 0; ig < p.segsPrep2; ig++) {
    segmentTimepoints(tp, t, ts);
  }
  t += p.TE;
  for (Index ig = 0; ig < (p.segsPerPrep - p.segsPrep2); ig++) {
    segmentTimepoints(tp, t, ts);
  }
  if (tp != traces()) { throw Log::Failure("Sim", "Programmer error"); }
  return ts;
}

} // namespace rl
