#include "sim-prep.h"

#include "unsupported/Eigen/MatrixFunctions"

namespace Sim {

Result Simple(
    Parameter const T1p,
    Parameter const betap,
    Parameter const B1p,
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
  log.info(FMT_STRING("{} values of T1 from {} to {}s"), T1p.N, T1p.lo, T1p.hi);
  log.info(FMT_STRING("{} values of β from {} to {}"), betap.N, betap.lo, betap.hi);
  log.info(FMT_STRING("{} values of B1 from {} to {}"), B1p.N, B1p.lo, B1p.hi);
  long totalN = T1p.N * betap.N * B1p.N;
  Eigen::MatrixXf sims(totalN, seq.sps); // SVD expects observations in rows
  Eigen::MatrixXf parameters(totalN, 4);

  long row = 0;
  for (long iB1 = 0; iB1 < B1p.N; iB1++) {
    float const B1 = B1p.value(iB1);
    float const cosa = cos(B1 * seq.alpha * M_PI / 180.f);
    float const sina = sin(B1 * seq.alpha * M_PI / 180.f);
    Eigen::Matrix2f A;
    A << cosa, 0.f, 0.f, 1.f;
    for (long it = 0; it < T1p.N; it++) {
      // Set up matrices
      float const T1 = T1p.value(it);
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

      for (long ib = 0; ib < betap.N; ib++) {
        float const beta = betap.value(ib);
        Eigen::Matrix2f B;
        B << beta, 0.f, 0.f, 1.f;

        // Get steady state after prep-pulse
        Eigen::Matrix2f const SS = (Ei * B * Er * (E1 * A).pow(seq.sps));
        float const Mz_ss = SS(0, 1) / (1.f - SS(0, 0));

        // Now fill in dynamic
        Eigen::Vector2f Mz{Mz_ss, 1.f};
        for (long ii = 0; ii < seq.sps; ii++) {
          sims(row, ii) = Mz(0) * sina;
          Mz = A * Mz;
          Mz = E1 * Mz;
        }
        parameters(row, 0) = Mz_ss;
        parameters(row, 1) = T1;
        parameters(row, 2) = beta;
        parameters(row, 3) = B1;
        row++;
      }
    }
  }
  return {sims, parameters};
}

} // namespace Sim
