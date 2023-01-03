#include "parameter.hpp"

#include "../log.hpp"
#include <random>

namespace rl::Parameters {

auto T1(Index const nS) -> Eigen::ArrayXXf
{
  float const R1lo = 1.f / 0.25f;
  float const R1hi = 1.f / 5.0f;
  auto const R1s = Eigen::ArrayXf::LinSpaced(nS, R1lo, R1hi);
  return 1.f / R1s.transpose();
}

auto T1T2(Index const nS) -> Eigen::ArrayXXf
{
  float const R1lo = 1.f / 0.25f;
  float const R1hi = 1.f / 4.3f;
  float const R2lo = 1.f / 0.02f;
  float const R2hi = 1.f / 3.f;
  Index const nT = std::floor(std::sqrt(nS));
  auto const R1s = Eigen::ArrayXf::LinSpaced(nT, R1lo, R1hi);
  auto const R2s = Eigen::ArrayXf::LinSpaced(nT, R2lo, R2hi);
  Index nAct = 0;
  Eigen::ArrayXXf p(2, nS);
  for (Index i1 = 0; i1 < nT; i1++) {
    for (Index i2 = 0; i2 < nT; i2++) {
      if (R2s(i2) > R1s(i1)) {
        p(0, nAct) = 1.f / R1s(i1);
        p(1, nAct) = 1.f / R2s(i2);
        nAct++;
      }
    }
  }
  p.conservativeResize(2, nAct);
  return p;
}

auto T1η(Index const nS) -> Eigen::ArrayXXf
{
  float const R1lo = 1.f / 0.25f;
  float const R1hi = 1.f / 4.3f;
  float const ηlo = 1.f / 0.9f;
  float const ηhi = 1.f / 1.f;
  Index const nT = std::floor(std::sqrt(nS));
  auto const R1s = Eigen::ArrayXf::LinSpaced(nT, R1lo, R1hi);
  auto const ηs = Eigen::ArrayXf::LinSpaced(nT, ηlo, ηhi);
  Index nAct = 0;
  Eigen::ArrayXXf p(2, nS);
  for (Index i1 = 0; i1 < nT; i1++) {
    for (Index iη = 0; iη < nT; iη++) {
      p(0, nAct) = 1.f / R1s(i1);
      p(1, nAct) = ηs(iη);
      nAct++;
    }
  }
  p.conservativeResize(2, nAct);
  return p;
}

} // namespace rl::Parameters
