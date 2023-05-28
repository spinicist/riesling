#include "dir.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

DIR::DIR(Settings const s)
  : Sequence{s}
{
  Log::Print(
    "DIR simulation TI1 {} Trec {} ⍺ {} TR {} SPG {}", settings.TI, settings.Trec, settings.alpha, settings.TR, settings.spg);
}

Index DIR::length() const { return settings.spg * settings.gps; }

Eigen::ArrayXXf DIR::parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const
{
  return Parameters::T1B1η(nsamp, lo, hi);
}

Eigen::ArrayXf DIR::simulate(Eigen::ArrayXf const &p) const
{
  float const T1 = p(0);
  float const B1 = p(1);
  float const η = p(2);
  Eigen::ArrayXf dynamic(settings.spg * settings.gps);

  Eigen::Matrix2f inv;
  inv << -η, 0.f, 0.f, 1.f;

  float const R1 = 1.f / T1;
  Eigen::Matrix2f E1, Einv, Einv2, Eramp, Essi, Erec, Esat;
  float const e1 = exp(-R1 * settings.TR);
  float const einv = exp(-R1 * settings.TI);
  float const eramp = exp(-R1 * settings.Tramp);
  float const essi = exp(-R1 * settings.Tssi);
  float const erec = exp(-R1 * settings.Trec);
  float const esat = exp(-R1 * settings.Tsat);
  E1 << e1, 1.f - e1, 0.f, 1.f;
  Einv << einv, 1.f - einv, 0.f, 1.f;
  Eramp << eramp, 1.f - eramp, 0.f, 1.f;
  Essi << essi, 1.f - essi, 0.f, 1.f;
  Erec << erec, 1.f - erec, 0.f, 1.f;
  Esat << esat, 1.f - esat, 0.f, 1.f;

  float const cosa = cos(B1 * settings.alpha * M_PI / 180.f);
  float const sina = sin(B1 * settings.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const grp = (Essi * Eramp * (E1 * A).pow(settings.spg) * Eramp * Esat);
  Eigen::Matrix2f const SS =
    Einv * inv * Erec * grp.pow(settings.gps - settings.gprep2) * Einv * inv * grp.pow(settings.gprep2);
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  for (Index ig = 0; ig < settings.gprep2; ig++) {
    Mz = Esat * Mz;
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  Mz = Einv * inv * Mz;
  for (Index ig = 0; ig < settings.gps - settings.gprep2; ig++) {
    Mz = Esat * Mz;
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != settings.spg * settings.gps) {
    Log::Fail("Programmer error");
  }
  return dynamic;
}

DIR2::DIR2(Settings const s)
  : Sequence{s}
{
  Log::Print(
    "DIR2 simulation TI1 {} Trec {} ⍺ {} TR {} SPG {} TE {}",
    settings.TI,
    settings.Trec,
    settings.alpha,
    settings.TR,
    settings.spg,
    settings.TE);
}

Index DIR2::length() const { return settings.spg * settings.gps; }

Eigen::ArrayXXf DIR2::parameters(Index const nsamp, std::vector<float> lo, std::vector<float> hi) const
{
  return Parameters::T1T2PD(nsamp, lo, hi);
}

Eigen::ArrayXf DIR2::simulate(Eigen::ArrayXf const &p) const
{
  float const T1 = p(0);
  float const T2 = p(1);
  float const PD = p(2);
  float const B1 = 0.7;
  float const η = 1.0f;
  Eigen::ArrayXf dynamic(settings.spg * settings.gps);

  Eigen::Matrix2f inv;
  inv << -η, 0.f, 0.f, 1.f;

  float const R1 = 1.f / T1;
  float const R2 = 1.f / T2;
  Eigen::Matrix2f E1, E2, Einv, Eramp, Essi, Erec, Esat;
  float const e1 = exp(-R1 * settings.TR);
  float const e2 = exp(-R2 * settings.TE);
  float const einv = exp(-R1 * settings.TI);
  float const eramp = exp(-R1 * settings.Tramp);
  float const essi = exp(-R1 * settings.Tssi);
  float const erec = exp(-R1 * settings.Trec);
  float const esat = exp(-R1 * settings.Tsat);
  E1 << e1, 1.f - e1, 0.f, 1.f;
  Einv << einv, 1.f - einv, 0.f, 1.f;
  E2 << -e2, 0.f, 0.f, 1.f;
  Eramp << eramp, 1.f - eramp, 0.f, 1.f;
  Essi << essi, 1.f - essi, 0.f, 1.f;
  Erec << erec, 1.f - erec, 0.f, 1.f;
  Esat << esat, 1.f - esat, 0.f, 1.f;

  float const cosa = cos(B1 * settings.alpha * M_PI / 180.f);
  float const sina = sin(B1 * settings.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const grp = (Essi * Eramp * (E1 * A).pow(settings.spg) * Eramp * Esat);
  Eigen::Matrix2f const SS = Einv * inv * Erec * grp.pow(settings.gps - settings.gprep2) * E2 * grp.pow(settings.gprep2);
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  Mz *= PD;
  for (Index ig = 0; ig < settings.gprep2; ig++) {
    Mz = Esat * Mz;
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  Mz = E2 * Mz;
  for (Index ig = 0; ig < settings.gps - settings.gprep2; ig++) {
    Mz = Esat * Mz;
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != settings.spg * settings.gps) {
    Log::Fail("Programmer error");
  }
  return dynamic;
}

Prep2::Prep2(Settings const s)
  : Sequence{s}
{
  Log::Print(
    "Prep2 simulation Trec {} ⍺ {} TR {} SPG {} TE {}", settings.Trec, settings.alpha, settings.TR, settings.spg, settings.TE);
}

Index Prep2::length() const { return settings.spg * settings.gps; }

Eigen::ArrayXXf Prep2::parameters(Index const nS, std::vector<float> lo, std::vector<float> hi) const
{
  float const PDlo = lo[0];
  float const PDhi = hi[0];
  float const R1lo = 1.f / lo[1];
  float const R1hi = 1.f / hi[1];
  float const β1lo = lo[2];
  float const β1hi = hi[2];
  float const β2lo = lo[3];
  float const β2hi = hi[3];
  Index const nT = std::floor(std::pow(nS / 10, 1. / 3.));
  auto const PDs = Eigen::ArrayXf::LinSpaced(10, PDlo, PDhi);
  auto const R1s = Eigen::ArrayXf::LinSpaced(nT, R1lo, R1hi);
  auto const β1s = Eigen::ArrayXf::LinSpaced(nT, β1lo, β1hi);
  auto const β2s = Eigen::ArrayXf::LinSpaced(nT, β2lo, β2hi);
  Index nAct = 0;
  Eigen::ArrayXXf p(4, nS);
  for (Index i4 = 0; i4 < 10; i4++) {
    for (Index i3 = 0; i3 < nT; i3++) {
      for (Index i2 = 0; i2 < nT; i2++) {
        for (Index i1 = 0; i1 < nT; i1++) {
          p(0, nAct) = PDs(i4);
          p(1, nAct) = 1.f / R1s(i3);
          p(0, nAct) = β1s(i2);
          p(0, nAct) = β2s(i3);
          nAct++;
        }
      }
    }
  }
  p.conservativeResize(4, nAct);
  return p;
}

Eigen::ArrayXf Prep2::simulate(Eigen::ArrayXf const &p) const
{
  float const PD = p(0);
  float const T1 = p(1);
  float const β1 = p(2);
  float const β2 = p(3);
  Eigen::ArrayXf dynamic(settings.spg * settings.gps);

  Eigen::Matrix2f prep1, prep2;
  prep1 << β1, 0.f, 0.f, 1.f;
  prep2 << β2, 0.f, 0.f, 1.f;

  float const R1 = 1.f / T1;
  Eigen::Matrix2f E1, Eramp, Essi, Erec;
  float const e1 = exp(-R1 * settings.TR);
  float const eramp = exp(-R1 * settings.Tramp);
  float const essi = exp(-R1 * settings.Tssi);
  float const erec = exp(-R1 * settings.Trec);
  E1 << e1, 1.f - e1, 0.f, 1.f;
  Eramp << eramp, 1.f - eramp, 0.f, 1.f;
  Essi << essi, 1.f - essi, 0.f, 1.f;
  Erec << erec, 1.f - erec, 0.f, 1.f;
  float const cosa = cos(settings.alpha * M_PI / 180.f);
  float const sina = sin(settings.alpha * M_PI / 180.f);

  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const grp = (Essi * Eramp * (E1 * A).pow(settings.spg + settings.spoil) * Eramp);
  Eigen::Matrix2f const SS = prep1 * Erec * grp.pow(settings.gps - settings.gprep2) * prep2 * grp.pow(settings.gprep2);
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  Mz *= PD;
  for (Index ig = 0; ig < settings.gprep2; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  Mz = prep2 * Mz;
  for (Index ig = 0; ig < settings.gps - settings.gprep2; ig++) {
    Mz = Eramp * Mz;
    for (Index ii = 0; ii < settings.spoil; ii++) {
      Mz = E1 * A * Mz;
    }
    for (Index ii = 0; ii < settings.spg; ii++) {
      dynamic(tp++) = Mz(0) * sina;
      Mz = E1 * A * Mz;
    }
    Mz = Essi * Eramp * Mz;
  }
  if (tp != settings.spg * settings.gps) {
    Log::Fail("Programmer error");
  }
  return dynamic;
}

} // namespace rl
