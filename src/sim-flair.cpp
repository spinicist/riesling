#include "sim-mupa.h"

#include "threads.h"
#include "unsupported/Eigen/MatrixFunctions"

namespace Sim {

Result FLAIR(
  Parameter const T1p,
  Parameter const T2p,
  Parameter const B1p,
  Sequence const seq,
  long const nRand,
  Log &log)
{
  log.info("FLAIR-ZTE simulation");
  log.info(
    FMT_STRING("SPS {}, FA {}, TR {}s, TI {}s, Trec {}s, TE {}s"),
    seq.sps,
    seq.alpha,
    seq.TR,
    seq.TI,
    seq.Trec,
    seq.TE);
  ParameterGenerator<3> gen({T1p, T2p, B1p});
  long totalN = (nRand > 0) ? nRand : gen.totalN();
  if (nRand > 0) {
    log.info(FMT_STRING("Random values of T1 from {} to {}s"), T1p.lo, T1p.hi);
    log.info(FMT_STRING("Random values of T2 from {} to {}"), T2p.lo, T2p.hi);
    log.info(FMT_STRING("Random values of B1 from {} to {}"), B1p.lo, B1p.hi);
  } else {
    log.info(FMT_STRING("{} values of T1 from {} to {}s"), T1p.N, T1p.lo, T1p.hi);
    log.info(FMT_STRING("{} values of T2p from {} to {}"), T2p.N, T2p.lo, T2p.hi);
    log.info(FMT_STRING("{} values of B1 from {} to {}"), B1p.N, B1p.lo, B1p.hi);
  }
  log.info(FMT_STRING("{} total values"), totalN);

  Result result;
  result.dynamics.resize(totalN, seq.sps);
  result.parameters.resize(totalN, 3);
  result.Mz_ss.resize(totalN);

  Eigen::Matrix2f inv;
  inv << -1.f, 0.f, 0.f, 1.f;
  auto task = [&](long const lo, long const hi, long const ti) {
    for (long ip = lo; ip < hi; ip++) {
      log.progress(ip, lo, hi);
      // Set up matrices
      auto const P = (nRand > 0) ? gen.rand() : gen.values(ip);
      float const T1 = P(0);
      float const T2 = P(1);
      float const B1 = P(2);

      float const R1 = 1.f / T1;
      Eigen::Matrix2f E1, Eramp, Essi, Ei, Er, Erec;
      float const e1 = exp(-R1 * seq.TR);
      float const eramp = exp(-R1 * seq.Tramp);
      float const essi = exp(-R1 * seq.Tssi);
      float const einv = exp(-R1 * seq.TI);
      float const erec = exp(-R1 * seq.Trec);
      E1 << e1, 1 - e1, 0.f, 1.f;
      Eramp << eramp, 1 - eramp, 0.f, 1.f;
      Essi << essi, 1 - essi, 0.f, 1.f;
      Ei << einv, 1 - einv, 0.f, 1.f;
      Erec << erec, 1 - erec, 0.f, 1.f;

      float const R2 = 1.f / T2;
      Eigen::Matrix2f E2;
      E2 << exp(-R2 * seq.TE), 0.f, 0.f, 1.f;

      float const cosa = cos(B1 * seq.alpha * M_PI / 180.f);
      float const sina = sin(B1 * seq.alpha * M_PI / 180.f);

      Eigen::Matrix2f A;
      A << cosa, 0.f, 0.f, 1.f;

      // Get steady state before first read-out
      Eigen::Matrix2f const SS = Eramp * E2 * Ei * inv * Erec * Eramp * (E1 * A).pow(seq.sps);
      float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

      // Now fill in dynamic
      long tp = 0;
      Eigen::Vector2f Mz{m_ss, 1.f};
      for (long ii = 0; ii < seq.sps; ii++) {
        result.dynamics(ip, tp++) = Mz(0) * sina;
        Mz = E1 * A * Mz;
      }
      if (tp != seq.sps) {
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
