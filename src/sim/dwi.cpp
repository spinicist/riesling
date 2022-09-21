#include "dwi.hpp"

#include "unsupported/Eigen/MatrixFunctions"

namespace rl {

DWI::DWI(Settings const &s)
  : Sequence{s}
{
}

Index DWI::length() const
{
  return 4 * settings.spg * settings.gps;
}

Eigen::ArrayXXf DWI::parameters(Index const nsamp) const
{
  Parameter gamma{0.f, 0.f, -M_PI, M_PI, true};
  Parameter Dwm{750e-6f, 50e-6f, 500e-6f, 1000e-6f}, Dgm{850e-6f, 100e-6f, 600e-6f, 1100e-6f},
    Dcsf{3200e-6f, 200e-6f, 2800e-6f, 3500e-6f};
  Tissues tissues(
    {Tissue{{T1wm, T2wm, Dwm, gamma}}, Tissue{{T1gm, T2gm, Dgm, gamma}}, Tissue{{T1csf, T2csf, Dcsf, gamma}}});
  return tissues.values(nsamp);
}

Eigen::ArrayXf DWI::simulate(Eigen::ArrayXf const &p) const
{
  Eigen::ArrayXf dynamic(4 * settings.spg * settings.gps);
  float const T1 = p(0);
  float const T2 = p(1);
  float const D = p(2);
  float const gamma = p(3);
  float const R1 = 1.f / T1;
  float const e1 = exp(-R1 * settings.TR);
  float const eramp = exp(-R1 * settings.Tramp);
  float const essi = exp(-R1 * settings.Tssi);
  float const erec = exp(-R1 * settings.Trec);
  Eigen::Matrix2f E1, Eramp, Essi, Einv, Erec;
  E1 << e1, 1 - e1, 0.f, 1.f;
  Eramp << eramp, 1 - eramp, 0.f, 1.f;
  Essi << essi, 1 - essi, 0.f, 1.f;
  Erec << erec, 1 - erec, 0.f, 1.f;

  float const cosa = cos(settings.alpha * M_PI / 180.f);
  float const sina = sin(settings.alpha * M_PI / 180.f);
  Eigen::Matrix2f A;
  A << cosa, 0.f, 0.f, 1.f;

  float const R2 = 1.f / T2;
  std::array<Eigen::Matrix2f, 4> preps;
  float pinc = M_PI / 2.f;
  float const beta = exp(-R2 * settings.TE - settings.bval * D);
  // These are arranged this way to fit with where we find the steady-state below
  preps[0] << beta * cos(gamma + pinc), 0.f, 0.f, 1.f;
  preps[1] << beta * cos(gamma + pinc * 2.f), 0.f, 0.f, 1.f;
  preps[2] << beta * cos(gamma + pinc * 3.f), 0.f, 0.f, 1.f;
  preps[3] << beta * cos(gamma), 0.f, 0.f, 1.f;

  // Get steady state after prep-pulse for first segment
  Eigen::Matrix2f const seg = Erec * (Essi * Eramp * (E1 * A).pow(settings.spg) * Eramp).pow(settings.gps);
  Eigen::Matrix2f SS = Eigen::Matrix2f::Identity();
  for (int ii = 0; ii < 4; ii++) {
    SS = preps[ii] * seg * SS;
  }
  float const m_ss = SS(0, 1) / (1.f - SS(0, 0));

  // Now fill in dynamic
  Index tp = 0;
  Eigen::Vector2f Mz{m_ss, 1.f};
  for (int ip = 0; ip < 4; ip++) {
    for (Index ig = 0; ig < settings.gps; ig++) {
      Mz = Eramp * Mz;
      for (Index ii = 0; ii < settings.spg; ii++) {
        dynamic(tp++) = Mz(0) * sina;
        Mz = E1 * A * Mz;
      }
      Mz = Essi * Eramp * Mz;
    }
    Mz = preps[ip] * Mz;
  }

  if (tp != (4 * settings.spg * settings.gps)) {
    Log::Fail("Programmer error");
  }

  return dynamic;
}

} // namespace rl
