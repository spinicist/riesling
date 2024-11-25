#include "ir.hpp"

#include "../log.hpp"
#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

IR::IR(Pars const &s, bool const pt)
  : SegmentedZTE{s, pt}
{
}

auto IR::nTissueParameters() const -> Index {return 5; }

auto IR::simulate(Eigen::ArrayXf const &pars) const -> Cx2
{
  if (pars.size() != nTissueParameters()) { throw Log::Failure("Sim", "Need {} parameters T1 T2 Δf Q", nTissueParameters()); }
  float const R1 = 1.f / pars(0);
  float const R2 = 1.f / pars(1);
  float const Δf = pars(2);
  float const Q = pars(3);
  float const B1 = pars(4);

  Eigen::Matrix2f const V = inv(Q), Er = E(R1, p.Trec), seg = Eseg(R1, B1);
  Eigen::Matrix2f const SS = V * seg.pow(p.segsPerPrep) * Er;
  float const           m_ss = SS(0, 1) / (1.f - SS(0, 0));

  Eigen::Vector2f Mz{m_ss, 1.f};
  Cx1             s0(traces());
  Index           tp = 0;
  for (Index ig = 0; ig < p.segsPerPrep; ig++) {
    segment(tp, Mz, s0, R1, B1);
  }
  if (tp != traces()) { throw Log::Failure("Sim", "Programmer error"); }
  return readout(pars(1), Δf).contract(s0, Eigen::array<Eigen::IndexPair<Index>, 0>());
}

auto IR::timepoints() const -> Re1
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
