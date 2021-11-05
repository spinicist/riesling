#include "sim-eddy.h"

#include "threads.h"
#include "unsupported/Eigen/MatrixFunctions"

namespace Sim {

Result Eddy(
    Parameter const T1p,
    Parameter const betap,
    Parameter const gammap,
    Parameter const B1p,
    Sequence const seq,
    long const nRand,
    Log &log)
{
  log.info("Eddy Current MP-ZTE simulation");
  log.info(
      FMT_STRING("SPS {}, FA {}, TR {}s, Trec {}s, Tramp {}s, Tssi {}s"),
      seq.sps,
      seq.alpha,
      seq.TR,
      seq.Trec,
      seq.Tramp,
      seq.Tssi);
  log.info(FMT_STRING("{} values of T1 from {} to {}s"), T1p.N, T1p.lo, T1p.hi);
  log.info(FMT_STRING("{} values of β from {} to {}"), betap.N, betap.lo, betap.hi);
  log.info(FMT_STRING("{} values of ɣ from {} to {}"), gammap.N, gammap.lo, gammap.hi);
  log.info(FMT_STRING("{} values of B1 from {} to {}"), B1p.N, B1p.lo, B1p.hi);
  ParameterGenerator<4> gen({T1p, betap, gammap, B1p});
  long totalN = (nRand > 0) ? nRand : gen.totalN();
  Result result;
  result.dynamics.resize(totalN, 4 * seq.sps);
  result.parameters.resize(totalN, 4);
  result.Mz_ss.resize(totalN);

  auto task = [&](long const lo, long const hi, long const ti) {
    for (long ip = lo; ip < hi; ip++) {
      log.progress(ip, lo, hi);
      auto const P = (nRand > 0) ? gen.rand() : gen.values(ip);
      // Set up matrices
      float const T1 = P(0);
      float const R1 = 1.f / T1;
      float const e1 = exp(-R1 * seq.TR);
      float const eramp = exp(-R1 * seq.Tramp);
      float const essi = exp(-R1 * seq.Tssi);
      float const erec = exp(-R1 * seq.Trec);
      Eigen::Matrix2f E1, Eramp, Essi, Einv, Erec;
      E1 << e1, 1 - e1, 0.f, 1.f;
      Eramp << eramp, 1 - eramp, 0.f, 1.f;
      Essi << essi, 1 - essi, 0.f, 1.f;
      Erec << erec, 1 - erec, 0.f, 1.f;

      float const B1 = P(1);
      float const cosa = cos(B1 * seq.alpha * M_PI / 180.f);
      float const sina = sin(B1 * seq.alpha * M_PI / 180.f);
      Eigen::Matrix2f A;
      A << cosa, 0.f, 0.f, 1.f;

      float const beta = P(3);
      float const gamma = P(2);
      Eigen::Matrix2f PC0, PC1, PC2, PC3;
      float pinc = M_PI / 2.f;
      PC0 << beta * cos(gamma), 0.f, 0.f, 1.f;
      PC1 << beta * cos(gamma + pinc), 0.f, 0.f, 1.f;
      PC2 << beta * cos(gamma + pinc * 2.f), 0.f, 0.f, 1.f;
      PC3 << beta * cos(gamma + pinc * 3.f), 0.f, 0.f, 1.f;

      // Get steady state after prep-pulse for first segment
      Eigen::Matrix2f const seg = Erec * Essi * Eramp * (E1 * A).pow(seq.sps) * Eramp;
      Eigen::Matrix2f const SS = PC0 * seg * PC3 * seg * PC2 * seg * PC1 * seg;
      float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

      // Now fill in dynamic
      long col = 0;
      Eigen::Vector2f Mz{m_ss, 1.f};
      Mz = Eramp * Mz;
      for (long ii = 0; ii < seq.sps; ii++) {
        result.dynamics(ip, col++) = Mz(0) * sina;
        Mz = E1 * A * Mz;
      }
      Mz = Eramp * PC1 * Erec * Essi * Eramp * Mz;
      for (long ii = 0; ii < seq.sps; ii++) {
        result.dynamics(ip, col++) = Mz(0) * sina;
        Mz = E1 * A * Mz;
      }
      Mz = Eramp * PC2 * Erec * Essi * Eramp * Mz;
      for (long ii = 0; ii < seq.sps; ii++) {
        result.dynamics(ip, col++) = Mz(0) * sina;
        Mz = E1 * A * Mz;
      }
      Mz = Eramp * PC3 * Erec * Essi * Eramp * Mz;
      for (long ii = 0; ii < seq.sps; ii++) {
        result.dynamics(ip, col++) = Mz(0) * sina;
        Mz = E1 * A * Mz;
      }
      if (col != (4 * seq.sps)) {
        Log::Fail("Programmer error");
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
