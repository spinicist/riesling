#include "sim-eddy.h"

#include "threads.h"
#include "unsupported/Eigen/MatrixFunctions"

namespace Sim {

SimResult Diffusion(
    long const nT1,
    float const T1Lo,
    float const T1Hi,
    long const nbeta,
    float const betaLo,
    float const betaHi,
    long const ngamma,
    Sequence const seq,
    Log &log)
{
  log.info("Eddy Current MP-ZTE simulation");
  log.info(
      FMT_STRING("SPS {}, FA {}, TR {}s, TI {}s, Trec {}s"),
      seq.sps,
      seq.alpha,
      seq.TR,
      seq.TI,
      seq.Trec);
  log.info(FMT_STRING("{} values of T1 from {} to {}s"), nT1, T1Lo, T1Hi);
  log.info(FMT_STRING("{} values of β from {} to {}"), nbeta, betaLo, betaHi);
  log.info(FMT_STRING("{} values of ɣ from 0 to 2π"), ngamma);
  long totalN = nT1 * nbeta * ngamma;
  Eigen::MatrixXf sims(totalN, 4 * seq.sps); // SVD expects observations in rows
  Eigen::MatrixXf parameters(totalN, 3);

  float const cosa = cos(seq.alpha * M_PI / 180.f);
  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;
  auto task = [&](long const lo, long const hi, long const ti) {
    long row = lo * ngamma * nbeta;
    for (long it = lo; it < hi; it++) {
      log.progress(it, lo, hi);
      // Set up matrices
      float const fit = it / (nT1 - 1.f);
      float const T1 = T1Lo * std::pow(T1Hi / T1Lo, fit);
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
      for (long ib = 0; ib < nbeta; ib++) {
        float const fib = ib / (nbeta - 1.f);
        float const beta = betaLo * std::pow(betaHi / betaLo, fib);
        Eigen::Matrix2f B;
        B << beta, 0.f, 0.f, 1.f;
        for (long ig = 0; ig < ngamma; ig++) {
          float const gamma = ig * 2. * M_PI / ngamma;
          Eigen::Matrix2f PC0, PC1, PC2, PC3;
          PC0 << sin(gamma), 0.f, 0.f, 1.f;
          PC1 << sin(gamma + M_PI / 2.f), 0.f, 0.f, 1.f;
          PC2 << sin(gamma + M_PI), 0.f, 0.f, 1.f;
          PC3 << sin(gamma + 3.f * M_PI / 2.f), 0.f, 0.f, 1.f;

          // Get steady state after prep-pulse for first segment
          Eigen::Matrix2f const seg = B * Er * (E1 * A).pow(seq.sps);
          Eigen::Matrix2f const SS =
              Ei * PC0 * seg * Ei * PC3 * seg * Ei * PC2 * seg * Ei * PC1 * seg;
          float const Mz_ss = SS(0, 1) / (1.f - SS(0, 0));

          // Now fill in dynamic
          long col = 0;
          Eigen::Vector2f Mz{Mz_ss, 1.f};
          for (long ii = 0; ii < seq.sps; ii++) {
            Mz = A * Mz;
            sims(row, col++) = Mz(0);
            Mz = E1 * Mz;
          }
          Mz = Ei * PC1 * B * Er * Mz;
          for (long ii = 0; ii < seq.sps; ii++) {
            Mz = A * Mz;
            sims(row, col++) = Mz(0);
            Mz = E1 * Mz;
          }
          Mz = Ei * PC2 * B * Er * Mz;
          for (long ii = 0; ii < seq.sps; ii++) {
            Mz = A * Mz;
            sims(row, col++) = Mz(0);
            Mz = E1 * Mz;
          }
          Mz = Ei * PC3 * B * Er * Mz;
          for (long ii = 0; ii < seq.sps; ii++) {
            Mz = A * Mz;
            sims(row, col++) = Mz(0);
            Mz = E1 * Mz;
          }
          if (col != (4 * seq.sps)) {
            Log::Fail("Programmer error");
          }
          parameters(row, 0) = T1;
          parameters(row, 1) = beta;
          parameters(row, 2) = gamma;
          row++;
        }
      }
    }
  };
  auto const start = log.now();
  Threads::RangeFor(task, nT1);
  log.info("Simulation took {}", log.toNow(start));
  return {sims, parameters};
}

} // namespace Sim
