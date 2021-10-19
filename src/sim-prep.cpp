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
    bool const betaLog,
    Sequence const seq,
    Log &log)
{
  log.info("Basic MP-ZTE simulation");
  log.info(
      FMT_STRING("SPS {}, FA {}, TR {}s, TI {}s, Trec {}s"),
      seq.sps,
      seq.alpha,
      seq.TR,
      seq.TI,
      seq.Trec);
  log.info(FMT_STRING("{} values of T1 from {} to {}s"), nT1, T1Lo, T1Hi);
  log.info(FMT_STRING("{} values of β from {} to {}"), nbeta, betaLo, betaHi);
  long totalN = nT1 * nbeta;
  Eigen::MatrixXf sims(totalN, seq.sps); // SVD expects observations in rows
  Eigen::MatrixXf parameters(totalN, 2);

  float const cosa = cos(seq.alpha * M_PI / 180.f);
  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;
  for (long it = 0; it < nT1; it++) {
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
      float const ref = 0.1;
      float const beta = betaLog ? (ref * std::pow((betaHi - betaLo) / ref, fib) - (ref - betaLo))
                                 : (betaLo + fib * (betaHi - betaLo));
      Eigen::Matrix2f B;
      B << beta, 0.f, 0.f, 1.f;

      // Get steady state after prep-pulse
      Eigen::Matrix2f const SS = (Ei * B * Er * (E1 * A).pow(seq.sps));
      float const Mz_ss = SS(0, 1) / (1.f - SS(0, 0));

      // Now fill in dynamic
      long const row = it * nbeta + ib;
      Eigen::Vector2f Mz{Mz_ss, 1.f};
      for (long ii = 0; ii < seq.sps; ii++) {
        Mz = A * Mz;
        sims(row, ii) = Mz(0);
        Mz = E1 * Mz;
      }
      parameters(row, 0) = T1;
      parameters(row, 1) = beta;
    }
  }
  return {sims, parameters};
}

SimResult PhaseCycled(
    long const nT1,
    float const T1Lo,
    float const T1Hi,
    long const nbeta,
    float const betaLo,
    float const betaHi,
    Sequence const seq,
    Log &log)
{
  log.info("Phase-cycled MP-ZTE simulation");
  log.info(
      FMT_STRING("SPS {}, FA {}, TR {}s, TI {}s, Trec {}s"),
      seq.sps,
      seq.alpha,
      seq.TR,
      seq.TI,
      seq.Trec);
  log.info(FMT_STRING("{} values of T1 from {} to {}s"), nT1, T1Lo, T1Hi);
  log.info(FMT_STRING("{} values of β from {} to {}"), nbeta, betaLo, betaHi);
  long totalN = nT1 * nbeta;
  Eigen::MatrixXf sims(totalN, 2 * seq.sps); // SVD expects observations in rows
  Eigen::MatrixXf parameters(totalN, 2);

  float const cosa = cos(seq.alpha * M_PI / 180.f);
  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;
  for (long it = 0; it < nT1; it++) {
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
      Eigen::Matrix2f B, Bi;
      B << beta, 0.f, 0.f, 1.f;
      Bi << -beta, 0.f, 0.f, 1.f;

      // Get steady state after prep-pulse
      Eigen::Matrix2f const SS =
          Ei * B * Er * (E1 * A).pow(seq.sps) * Ei * Bi * Er * (E1 * A).pow(seq.sps);
      float const Mz_ss = SS(0, 1) / (1.f - SS(0, 0));

      // Now fill in dynamic
      long const row = it * nbeta + ib;
      long col = 0;
      Eigen::Vector2f Mz{Mz_ss, 1.f};
      for (long ii = 0; ii < seq.sps; ii++) {
        Mz = A * Mz;
        sims(row, col++) = Mz(0);
        Mz = E1 * Mz;
      }
      Mz = Ei * Bi * Er * Mz;
      for (long ii = 0; ii < seq.sps; ii++) {
        Mz = A * Mz;
        sims(row, col++) = Mz(0);
        Mz = E1 * Mz;
      }
      parameters(row, 0) = T1;
      parameters(row, 1) = beta;
    }
  }
  return {sims, parameters};
}

} // namespace Sim
