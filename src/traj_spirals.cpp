#include "traj_spirals.h"
#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {
/* This implements the Archimedean spiral described in S. T. S. Wong and M. S. Roos, ‘A strategy for
 * sampling on a sphere applied to 3D selective RF pulse design’, Magnetic Resonance in Medicine,
 * vol. 32, no. 6, pp. 778–784, Dec. 1994, doi: 10.1002/mrm.1910320614.
 */
Re3 ArchimedeanSpiral(Index const nRead, Index const nSpoke)
{
  Re3 traj(3, nRead, nSpoke);
  Re1 read(nRead);
  for (Index ir = 0; ir < nRead; ir++) {
    read(ir) = (float)(ir) / (nRead - 1);
  }
  // Currently to do an outer product, you need to contract over empty indices
  Eigen::array<Eigen::IndexPair<Index>, 0> empty = {};
  Re1                                      endPoint(Sz1{3});

  // Slightly annoying as can't initialize a 1D Tensor with {} notation yet
  endPoint(0) = 0.f;
  endPoint(1) = 0.f;
  endPoint(2) = 1.f;
  traj.chip(0, 2) = endPoint.contract(read, empty);
  Index const half = nSpoke / 2;
  endPoint(0) = 1.f;
  endPoint(1) = 0.f;
  endPoint(2) = 0.f;
  traj.chip(half, 2) = endPoint.contract(read, empty);

  double const d_t = 1. / half;         // Change in theta
  double const c2 = nSpoke * 4. * M_PI; // The velocity squared
  double       t = 0.;
  double       phi = 0.;
  for (Index is = 1; is < half; is++) {
    t += d_t;
    double const cos_t2 = t * t;
    double const sin_t2 = 1. - cos_t2;
    double const sin_t = sqrt(sin_t2);
    double const d_phi = 0.5 * d_t * std::sqrt((1. / sin_t2) * (c2 - (1. / sin_t2)));
    phi += d_phi;
    endPoint(0) = std::cos(phi) * sin_t;
    endPoint(1) = std::sin(phi) * sin_t;
    endPoint(2) = t;
    traj.chip(half - is, 2) = endPoint.contract(read, empty);
    endPoint(0) = std::cos(phi) * sin_t;
    endPoint(1) = -std::sin(phi) * sin_t;
    endPoint(2) = -t;
    traj.chip(half + is, 2) = endPoint.contract(read, empty);
  }
  // Trajectory is stored between -0.5 and 0.5, so scale
  return 0.5f * traj;
}

Index Fib(Index n)
{
  if (n == 0) {
    return 0;
  } else if (n == 1) {
    return 1;
  } else if (n == 2) {
    return 1;
  } else {
    return (Fib(n - 1) + Fib(n - 2));
  }
}

Re3 Phyllotaxis(Index const nRead, Index const ntraces, Index const smoothness, Index const spi, bool const gm)
{
  if ((ntraces % spi) != 0) { Log::Fail("traces per interleave {} is not a divisor of total traces {}", spi, ntraces); }
  Index           nInterleaves = ntraces / spi;
  constexpr float phi_gold = 2.399963229728653;
  constexpr float phi_gm1 = 0.465571231876768;
  constexpr float phi_gm2 = 0.682327803828019;

  float dphi = phi_gold * Fib(smoothness);

  Re1 read(nRead);
  for (Index ir = 0; ir < nRead; ir++) {
    read(ir) = (float)(ir) / nRead;
  }
  // Currently to do an outer product, you need to contract over empty indices
  Eigen::array<Eigen::IndexPair<Index>, 0> empty = {};

  Re3 traj(3, nRead, ntraces);
  Re1 endPoint(3);
  for (Index ii = 0; ii < nInterleaves; ii++) {
    float const z_ii = (ii * 2.) / (ntraces - 1);
    float const phi_ii = (ii * phi_gold);
    for (Index is = 0; is < spi; is++) {
      float const z = (1 - (2. * nInterleaves * is) / (ntraces - 1.)) - z_ii;
      float const theta = acos(z);
      float const phi = (is * dphi) + phi_ii;
      endPoint(0) = sin(theta) * cos(phi);
      endPoint(1) = sin(theta) * sin(phi);
      endPoint(2) = z;
      if (gm) {
        float const gm_phi = 2 * M_PI * ii * phi_gm2;
        float const gm_theta = acos(fmod(ii * phi_gm1, 1.0) * 2.f - 1.f);
        Re2         rot(3, 3);
        rot(0, 0) = cos(gm_theta) * cos(gm_phi);
        rot(0, 1) = -sin(gm_phi);
        rot(0, 2) = cos(gm_phi) * sin(gm_theta);
        rot(1, 0) = cos(gm_theta) * sin(gm_phi);
        rot(1, 1) = cos(gm_phi);
        rot(1, 2) = sin(gm_theta) * sin(gm_phi);
        rot(2, 0) = -sin(gm_theta);
        rot(2, 1) = 0;
        rot(2, 2) = cos(gm_theta);
        endPoint = rot.contract(endPoint, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>()).eval();
      }
      traj.chip((ii * spi) + is, 2) = endPoint.contract(read, empty);
    }
  }
  traj = traj * traj.constant(0.5); // Scale to -0.5 to 0.5
  return traj;
}
} // namespace rl
