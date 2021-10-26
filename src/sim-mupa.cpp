#include "sim-mupa.h"

#include "threads.h"
#include "unsupported/Eigen/MatrixFunctions"

namespace Sim {

Result MUPA(
  Parameter const T1p,
  Parameter const T2p,
  Parameter const B1p,
  Sequence const seq,
  Log &log)
{
  log.info("MUPA MP-ZTE simulation");
  log.info(
    FMT_STRING("SPS {}, FA {}, TR {}s, TI {}s, Trec {}s"),
    seq.sps,
    seq.alpha,
    seq.TR,
    seq.TI,
    seq.Trec);
  log.info(FMT_STRING("{} values of T1 from {} to {}s"), T1p.N, T1p.lo, T1p.hi);
  log.info(FMT_STRING("{} values of T2 from {} to {}"), T2p.N, T2p.lo, T2p.hi);
  log.info(FMT_STRING("{} values of B1 from {} to {}"), B1p.N, B1p.lo, B1p.hi);
  long totalN = T1p.N * T2p.N * B1p.N;
  Eigen::MatrixXf sims(totalN, 4 * seq.sps); // SVD expects observations in rows
  Eigen::MatrixXf parameters(totalN, 3);

  Eigen::Matrix2f inv;
  inv << -1.f, 0.f,
          0.f, 1.f;
  auto task = [&](long const lo, long const hi, long const ti) {
    long row = lo * T2p.N * B1p.N;
    for (long it = lo; it < hi; it++) {
      log.progress(it, lo, hi);
      // Set up matrices
      float const T1 = T1p.value(it);
      float const R1 = 1.f / T1;
      float const e1 = exp(-R1 * seq.TR);
      Eigen::Matrix2f E1;
      E1 << e1, 1 - e1,
            0.f, 1.f;
      float const einv = exp(-R1 * seq.TI);
      Eigen::Matrix2f Ei;
      Ei << einv, 1 - einv,
            0.f, 1.f;
      float const erec = exp(-R1 * seq.Trec);
      Eigen::Matrix2f Er;
      Er << erec, 1 - erec,
            0.f, 1.f;

      for (long iB1 = 0; iB1 < B1p.N; iB1++) {
        float const B1 = B1p.value(iB1);
        float const cosa = cos(B1 * seq.alpha * M_PI / 180.f);
        Eigen::Matrix2f A;
        A << cosa, 0.f,
             0.f, 1.f;

        for (long it2 = 0; it2 < T2p.N; it2++) {
          float const T2 = T2p.value(it2);
          float const R2 = 1.f / T2;
          Eigen::Matrix2f E2;
          E2 << exp(-R2 * 0.05), 0.f,
                0.f, 1.f;

          // Get steady state after prep-pulse for first segment
          Eigen::Matrix2f const seg = (E1 * A).pow(seq.sps);
          Eigen::Matrix2f const SS =
              E2 * seg * seg * seg * Ei * inv * seg;
          float const Mz_ss = SS(0, 1) / (1.f - SS(0, 0));

          // Now fill in dynamic
          long col = 0;
          Eigen::Vector2f Mz{Mz_ss, 1.f};
          for (long ii = 0; ii < seq.sps; ii++) {
              Mz = A * Mz;
              sims(row, col++) = Mz(0);
              Mz = E1 * Mz;
          }
          Mz = Ei * inv * Mz;
          for (long is = 0; is < 3; is++) {
            for (long ii = 0; ii < seq.sps; ii++) {
                Mz = A * Mz;
                sims(row, col++) = Mz(0);
                Mz = E1 * Mz;
            }
          }
          if (col != (4 * seq.sps)) {
              Log::Fail("Programmer error");
          }
          parameters(row, 0) = T1;
          parameters(row, 1) = T2;
          parameters(row, 2) = B1;
          row++;
        }
      }
    }
  };
  auto const start = log.now();
  Threads::RangeFor(task, T1p.N);
  log.info("Simulation took {}", log.toNow(start));
  return {sims, parameters};
}

} // namespace Sim
