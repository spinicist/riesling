#include "sim-prep.h"

#include "unsupported/Eigen/MatrixFunctions"

namespace Sim {

SimResult Simple(
    long const nT1,
    float const T1Lo,
    float const T1Hi,
    long const nbeta,
    float const betaLo,
    float const betaHi,
    long const sps,
    float const alpha,
    float const TR,
    float const TI,
    float const Trec,
    Log &log)
{
  log.info("Basic MP-ZTE simulation");
  log.info(FMT_STRING("Seg length {}, FA {}, TR {}s, TI {}s, Trec {}s"), sps, alpha, TR, TI, Trec);
  log.info(FMT_STRING("{} values of T1 from {} to {}s"), nT1, T1Lo, T1Hi);
  log.info(FMT_STRING("{} values of β from {} to {}"), nbeta, betaLo, betaHi);
  long totalN = nT1 * nbeta;
  Eigen::MatrixXf sims(totalN, sps); // SVD expects observations in rows
  Eigen::MatrixXf parameters(totalN, 2);

  float const cosa = cos(alpha * M_PI / 180.f);
  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;
  for (long ib = 0; ib < nbeta; ib++) {
    float const fib = ib / (nbeta - 1.f);
    float const beta = betaLo * (1.f - fib) + betaHi * fib;
    Eigen::Matrix2f B;
    B << beta, 0.f, 0.f, 1.f;
    for (long it = 0; it < nT1; it++) {
      // Set up matrices
      float const fit = it / (nT1 - 1.f);
      float const R1 = 1.f / (T1Lo * (1.f - fit) + T1Hi * fit);
      float const e1 = exp(-R1 * TR);
      Eigen::Matrix2f E1;
      E1 << e1, 1 - e1, 0.f, 1.f;
      float const einv = exp(-R1 * TI);
      Eigen::Matrix2f Ei;
      Ei << einv, 1 - einv, 0.f, 1.f;
      float const erec = exp(-R1 * Trec);
      Eigen::Matrix2f Er;
      Er << erec, 1 - erec, 0.f, 1.f;

      // Get steady state after prep-pulse
      Eigen::Matrix2f const SS = (Ei * B * Er * (E1 * A).pow(sps));
      float const Mz_ss = SS(0, 1) / (1.f - SS(0, 0));

      // Now fill in dynamic
      long const row = ib * nT1 + it;
      Eigen::Vector2f Mz{Mz_ss, 1.f};
      for (long ii = 0; ii < sps; ii++) {
        Mz = A * Mz;
        sims(row, ii) = Mz(0);
        Mz = E1 * Mz;
      }
      parameters(row, 0) = beta;
      parameters(row, 1) = 1.f / R1;
    }
  }
  return {sims, parameters};
}

} // namespace Sim
