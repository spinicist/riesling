#include "mupa.h"

#include "threads.h"
#include "unsupported/Eigen/MatrixFunctions"

namespace Sim {

Result FLAIR(Parameter const T1p, Parameter const T2p, Sequence const seq, Index const nRand)
{
  Log::Print("FLAIR-ZTE simulation");
  Log::Print(
    FMT_STRING("SPS {}, GPS {}, FA {}, TR {}s, TE {}s, Trec {}s"),
    seq.sps,
    seq.gps,
    seq.alpha,
    seq.TR,
    seq.TE,
    seq.Trec);
  Index const spg = seq.sps / seq.gps; // Spokes per group
  Log::Print(FMT_STRING("{} values of T1 from {} to {}s"), T1p.N, T1p.lo, T1p.hi);
  Log::Print(FMT_STRING("{} values of T2 from {} to {}"), T2p.N, T2p.lo, T2p.hi);
  ParameterGenerator<2> gen(T1p, T2p);
  Index totalN = (nRand > 0) ? nRand : gen.totalN();
  Result result;
  result.dynamics.resize(totalN, seq.sps);
  result.parameters.resize(totalN, 2);
  result.Mz_ss.resize(totalN);

  Eigen::Matrix2f inv;
  inv << -1.f, 0.f, 0.f, 1.f;
  auto task = [&](Index const lo, Index const hi, Index const ti) {
    for (Index ip = lo; ip < hi; ip++) {
      Log::Progress(ip, lo, hi);
      // Set up matrices
      auto const P = (nRand > 0) ? gen.rand() : gen.values(ip);
      float const T1 = P(0);
      float const T2 = P(1);

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

      float const cosa = cos(seq.alpha * M_PI / 180.f);
      float const sina = sin(seq.alpha * M_PI / 180.f);

      Eigen::Matrix2f A;
      A << cosa, 0.f, 0.f, 1.f;

      // Get steady state before first read-out
      Eigen::Matrix2f const seg = (Essi * Eramp * (E1 * A).pow(spg) * Eramp).pow(seq.gps);
      Eigen::Matrix2f const SS = Essi * E2 * Ei * inv * Erec * seg;
      float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

      // Now fill in dynamic
      Index tp = 0;
      Eigen::Vector2f Mz{m_ss, 1.f};
      for (Index ig = 0; ig < seq.gps; ig++) {
        Mz = Eramp * Mz;
        for (Index ii = 0; ii < spg; ii++) {
          result.dynamics(ip, tp++) = Mz(0) * sina;
          Mz = E1 * A * Mz;
        }
        Mz = Essi * Eramp * Mz;
      }
      if (tp != seq.sps) {
        Log::Fail("Programmer error");
      }
      result.Mz_ss(ip) = m_ss;
      result.parameters.row(ip) = P;
    }
  };
  auto const start = Log::Now();
  Threads::RangeFor(task, totalN);
  Log::Print("Simulation took {}", Log::ToNow(start));
  return result;
}

} // namespace Sim
