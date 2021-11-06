#include "sim-prep.h"

#include "threads.h"
#include "unsupported/Eigen/MatrixFunctions"

namespace Sim {

Result
Simple(Parameter const T1p, Parameter const betap, Sequence const seq, long const nRand, Log &log)
{
  log.info("Basic MP-ZTE simulation");
  log.info(
    FMT_STRING("SPS {}, FA {}, TR {}s, TI {}s, Trec {}s"),
    seq.sps,
    seq.alpha,
    seq.TR,
    seq.TI,
    seq.Trec);

  ParameterGenerator<2> gen({T1p, betap});
  long totalN = (nRand > 0) ? nRand : gen.totalN();
  if (nRand > 0) {
    log.info(FMT_STRING("Random values of T1 from {} to {}s"), T1p.lo, T1p.hi);
    log.info(FMT_STRING("Random values of β from {} to {}"), betap.lo, betap.hi);
  } else {
    log.info(FMT_STRING("{} values of T1 from {} to {}s"), T1p.N, T1p.lo, T1p.hi);
    log.info(FMT_STRING("{} values of β from {} to {}"), betap.N, betap.lo, betap.hi);
  }
  log.info(FMT_STRING("{} total values"), totalN);

  Result result;
  result.dynamics.resize(totalN, seq.sps);
  result.parameters.resize(totalN, 2);
  result.Mz_ss.resize(totalN);

  auto task = [&](long const lo, long const hi, long const ti) {
    for (long ip = lo; ip < hi; ip++) {
      auto const P = (nRand > 0) ? gen.rand() : gen.values(ip);
      float const T1 = P(0);
      float const beta = P(1);

      // Set up matrices
      float const cosa = cos(seq.alpha * M_PI / 180.f);
      float const sina = sin(seq.alpha * M_PI / 180.f);
      Eigen::Matrix2f A;
      A << cosa, 0.f, 0.f, 1.f;

      float const R1 = 1.f / T1;
      float const e1 = exp(-R1 * seq.TR);
      float const einv = exp(-R1 * seq.TI);
      float const erec = exp(-R1 * seq.Trec);
      float const eramp = exp(-R1 * seq.Tramp);
      float const essi = exp(-R1 * seq.Tssi);
      Eigen::Matrix2f E1, Ei, Er, Eramp, Essi;
      E1 << e1, 1 - e1, 0.f, 1.f;
      Ei << einv, 1 - einv, 0.f, 1.f;
      Er << erec, 1 - erec, 0.f, 1.f;
      Eramp << eramp, 1 - eramp, 0.f, 1.f;
      Essi << essi, 1 - essi, 0.f, 1.f;

      Eigen::Matrix2f B;
      B << beta, 0.f, 0.f, 1.f;

      // Get steady state after prep-pulse
      Eigen::Matrix2f const SS = Eramp * Ei * B * Er * Eramp * (E1 * A).pow(seq.sps);
      float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

      // Now fill in dynamic
      Eigen::Vector2f Mz{m_ss, 1.f};
      for (long ii = 0; ii < seq.sps; ii++) {
        result.dynamics(ip, ii) = Mz(0) * sina;
        Mz = E1 * A * Mz;
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
