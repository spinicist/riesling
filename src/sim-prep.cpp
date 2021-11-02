#include "sim-prep.h"

#include "threads.h"
#include "unsupported/Eigen/MatrixFunctions"

namespace Sim {

Result Simple(
    Parameter const T1p, Parameter const betap, Parameter const B1p, Sequence const seq, Log &log)
{
  log.info("Basic MP-ZTE simulation");
  log.info(
      FMT_STRING("SPS {}, FA {}, TR {}s, TI {}s, Trec {}s"),
      seq.sps,
      seq.alpha,
      seq.TR,
      seq.TI,
      seq.Trec);
  log.info(FMT_STRING("{} values of T1 from {} to {}s"), T1p.N, T1p.lo, T1p.hi);
  log.info(FMT_STRING("{} values of β from {} to {}"), betap.N, betap.lo, betap.hi);
  log.info(FMT_STRING("{} values of B1 from {} to {}"), B1p.N, B1p.lo, B1p.hi);
  ParameterGenerator<3> gen({T1p, betap, B1p});
  long const totalN = gen.totalN();
  Result result;
  result.dynamics.resize(totalN, seq.sps);
  result.parameters.resize(totalN, 3);
  result.Mz_ss.resize(totalN);

  auto task = [&](long const lo, long const hi, long const ti) {
    for (long ip = lo; ip < hi; ip++) {
      auto const P = gen.values(ip);
      float const B1 = P(2);
      float const cosa = cos(B1 * seq.alpha * M_PI / 180.f);
      float const sina = sin(B1 * seq.alpha * M_PI / 180.f);
      Eigen::Matrix2f A;
      A << cosa, 0.f, 0.f, 1.f;

      // Set up matrices
      float const T1 = P(0);
      float const R1 = 1.f / T1;
      float const e1 = exp(-R1 * seq.TR);
      Eigen::Matrix2f E1;
      E1 << e1, 1 - e1, 0.f, 1.f;
      float const einv = exp(-R1 * seq.TI);
      Eigen::Matrix2f Ei;
      Ei << einv, 1 - einv, 0.f, 1.f;
      float const erec = exp(-R1 * seq.Trec);
      Eigen::Matrix2f Er;
      Er << erec, 1 - erec, 0.f, 1.f;

      float const beta = P(1);
      Eigen::Matrix2f B;
      B << beta, 0.f, 0.f, 1.f;

      // Get steady state after prep-pulse
      Eigen::Matrix2f const SS = (Ei * B * Er * (E1 * A).pow(seq.sps));
      float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

      // Now fill in dynamic
      Eigen::Vector2f Mz{m_ss, 1.f};
      for (long ii = 0; ii < seq.sps; ii++) {
        result.dynamics(ip, ii) = Mz(0) * sina;
        Mz = A * Mz;
        Mz = E1 * Mz;
      }
      result.Mz_ss(ip) = m_ss;
      result.parameters.row(ip) = P;
    }
  };
  auto const start = log.now();
  Threads::RangeFor(task, totalN);
  log.info("Simulation took {}", log.toNow(start));
  return result;
}

} // namespace Sim
