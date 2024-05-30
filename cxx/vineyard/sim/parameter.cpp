#include "parameter.hpp"

#include "log.hpp"
#include <random>

namespace rl::Parameters {

void CheckSizes(size_t const             N,
                std::vector<float> const defLo,
                std::vector<float> const defHi,
                std::vector<float>      &lo,
                std::vector<float>      &hi)
{
  if (lo.size() == 0) {
    lo = defLo;
  } else if (lo.size() != N) {
    Log::Fail("Low parameters must have {} values", N);
  }
  if (hi.size() == 0) {
    hi = defHi;
  } else if (hi.size() != N) {
    Log::Fail("High parameters must have {} values", N);
  }
}

auto T1(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf
{
  CheckSizes(1, {0.35f}, {4.3f}, lo, hi);
  float const R1lo = 1.f / lo[0];
  float const R1hi = 1.f / hi[0];
  auto const  R1s = Eigen::ArrayXf::LinSpaced(nS, R1lo, R1hi);
  return 1.f / R1s.transpose();
}

auto T1T2PD(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf
{
  CheckSizes(3, {0.6f, 0.04f, 0.7f}, {4.3f, 2.f, 1.3f}, lo, hi);
  float const     R1lo = 1.f / lo[0];
  float const     R1hi = 1.f / hi[0];
  float const     R2lo = 1.f / lo[1];
  float const     R2hi = 1.f / hi[1];
  float const     PDlo = lo[2];
  float const     PDhi = hi[2];
  Index const     nT = std::floor(std::pow(nS / 10, 1. / 2.));
  auto const      R1s = Eigen::ArrayXf::LinSpaced(nT, R1lo, R1hi);
  auto const      R2s = Eigen::ArrayXf::LinSpaced(nT, R2lo, R2hi);
  auto const      PDs = Eigen::ArrayXf::LinSpaced(10, PDlo, PDhi);
  Index           nAct = 0;
  Eigen::ArrayXXf p(3, nS);
  for (Index ib = 0; ib < 10; ib++) {
    for (Index i2 = 0; i2 < nT; i2++) {
      for (Index i1 = 0; i1 < nT; i1++) {
        if (R2s(i2) > R1s(i1)) {
          p(0, nAct) = 1.f / R1s(i1);
          p(1, nAct) = 1.f / R2s(i2);
          p(2, nAct) = PDs(ib);
          nAct++;
        }
      }
    }
  }
  p.conservativeResize(3, nAct);
  return p;
}

auto T1T2η(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf
{
  CheckSizes(3, {0.6f, 0.04f, 0.7f}, {4.3f, 2.f, 1.3f}, lo, hi);
  float const     R1lo = 1.f / lo[0];
  float const     R1hi = 1.f / hi[0];
  float const     R2lo = 1.f / lo[1];
  float const     R2hi = 1.f / hi[1];
  float const     ηlo = lo[2];
  float const     ηhi = hi[2];
  Index const     nη = 10;
  Index const     nT = std::floor(std::pow(nS / nη, 1. / 2.));
  Index const     nTot = nη * nT * nT;
  auto const      R1s = Eigen::ArrayXf::LinSpaced(nT, R1lo, R1hi);
  auto const      R2s = Eigen::ArrayXf::LinSpaced(nT, R2lo, R2hi);
  auto const      ηs = Eigen::ArrayXf::LinSpaced(10, ηlo, ηhi);
  Eigen::ArrayXXf p(3, nTot);
  Index           nAct = 0;
  for (Index i3 = 0; i3 < nη; i3++) {
    for (Index i2 = 0; i2 < nT; i2++) {
      for (Index i1 = 0; i1 < nT; i1++) {
        if (R2s(i2) > R1s(i1)) {
          p(0, nAct) = 1.f / R1s(i1);
          p(1, nAct) = 1.f / R2s(i2);
          p(2, nAct) = ηs(i3);
          nAct++;
        }
      }
    }
  }
  p.conservativeResize(3, nAct);
  return p;
}

auto T1B1(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf
{
  CheckSizes(2, {0.6f, 0.5f}, {4.3f, 1.5f}, lo, hi);
  float const     R1lo = 1.f / lo[0];
  float const     R1hi = 1.f / hi[0];
  float const     B1lo = lo[1];
  float const     B1hi = hi[1];
  Index const     nT = std::floor(std::pow(nS, 1. / 2.));
  auto const      R1s = Eigen::ArrayXf::LinSpaced(nT, R1lo, R1hi);
  auto const      B1s = Eigen::ArrayXf::LinSpaced(nT, B1lo, B1hi);
  Index           nAct = 0;
  Eigen::ArrayXXf p(2, nS);

  for (Index ib = 0; ib < nT; ib++) {
    for (Index i1 = 0; i1 < nT; i1++) {
      p(0, nAct) = 1.f / R1s(i1);
      p(1, nAct) = B1s(ib);
      nAct++;
    }
  }

  p.conservativeResize(2, nAct);
  return p;
}

auto T1β(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf
{
  CheckSizes(2, {0.6f, -1.f}, {4.3f, 1.f}, lo, hi);
  float const     R1lo = 1.f / lo[0];
  float const     R1hi = 1.f / hi[0];
  float const     βlo = lo[1];
  float const     βhi = hi[1];
  Index const     nT = std::floor(std::pow(nS, 1. / 2.));
  auto const      R1s = Eigen::ArrayXf::LinSpaced(nT, R1lo, R1hi);
  auto const      βs = Eigen::ArrayXf::LinSpaced(nT, βlo, βhi);
  Eigen::ArrayXXf p(2, nT * nT);

  Index ind = 0;
  for (Index ib = 0; ib < nT; ib++) {
    for (Index i1 = 0; i1 < nT; i1++) {
      p(0, ind) = 1.f / R1s(i1);
      p(1, ind) = βs(ib);
      ind++;
    }
  }
  return p;
}

auto T1β1β2(Index const nS, std::vector<float> lo, std::vector<float> hi) -> Eigen::ArrayXXf
{
  CheckSizes(3, {0.6f, -1.f, -1.f}, {4.3f, 1.f, 1.f}, lo, hi);
  float const R1lo = 1.f / lo[0];
  float const R1hi = 1.f / hi[0];
  float const β1lo = lo[1];
  float const β1hi = hi[1];
  float const β2lo = lo[2];
  float const β2hi = hi[2];
  Index const nβ = 20;
  Index const nT = std::floor((float)nS / nβ / nβ);
  auto const  R1s = Eigen::ArrayXf::LinSpaced(nT, R1lo, R1hi);
  auto const  β1s = Eigen::ArrayXf::LinSpaced(nβ, β1lo, β1hi);
  auto const  β2s = Eigen::ArrayXf::LinSpaced(nβ, β2lo, β2hi);

  Log::Print("nT {} R1 {}:{} β1 {}:{} β2 {}:{}", nT, R1lo, R1hi, β1lo, β1hi, β2lo, β2hi);

  Eigen::ArrayXXf p(3, nT * nβ * nβ);

  Index ind = 0;
  for (Index ib2 = 0; ib2 < nβ; ib2++) {
    for (Index ib1 = 0; ib1 < nβ; ib1++) {
      for (Index i1 = 0; i1 < nT; i1++) {
        p(0, ind) = 1.f / R1s(i1);
        p(1, ind) = β1s(ib1);
        p(2, ind) = β2s(ib2);
        ind++;
      }
    }
  }
  return p;
}

} // namespace rl::Parameters
