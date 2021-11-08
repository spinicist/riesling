#include "traj_spirals.h"
#include "log.h"

/* This implements the Archimedean spiral described in S. T. S. Wong and M. S. Roos, ‘A strategy for
 * sampling on a sphere applied to 3D selective RF pulse design’, Magnetic Resonance in Medicine,
 * vol. 32, no. 6, pp. 778–784, Dec. 1994, doi: 10.1002/mrm.1910320614.
 */

R3 ASpiral(long const nRead, long const nSpoke)
{
  R3 traj(3, nRead, nSpoke);
  R1 read(nRead);
  for (long ir = 0; ir < nRead; ir++) {
    read(ir) = (float)(ir) / nRead;
  }
  // Currently to do an outer product, you need to contract over empty indices
  Eigen::array<Eigen::IndexPair<long>, 0> empty = {};
  Point3 point;
  Eigen::TensorMap<R1> endPoint(point.data(), Sz1{3});

  // Slightly annoying as can't initialize a 1D Tensor with {} notation yet
  point = {0.f, 0.f, 1.f};
  traj.chip(0, 2) = endPoint.contract(read, empty);
  long const half = nSpoke / 2;
  point = {1.f, 0.f, 0.f};
  traj.chip(half, 2) = endPoint.contract(read, empty);

  float const d_t = 1.f / half;         // Change in theta
  float const c2 = nSpoke * 4.f * M_PI; // The velocity squared
  float t = 0.f;
  float phi = 0.f;
  for (long is = 1; is < half; is++) {
    t += d_t;
    float const cos_t2 = t * t;
    float const sin_t2 = 1 - cos_t2;
    float const sin_t = sqrt(sin_t2);
    float const d_phi = 0.5f * d_t * sqrt((1.f / sin_t2) * (c2 - (1.f / sin_t2)));
    phi += d_phi;
    point = {cos(phi) * sin_t, sin(phi) * sin_t, t};
    traj.chip(half - is, 2) = endPoint.contract(read, empty);
    point = {cos(phi) * sin_t, -sin(phi) * sin_t, -t};
    traj.chip(half + is, 2) = endPoint.contract(read, empty);
  }
  // Trajectory is stored between -0.5 and 0.5, so scale
  return 0.5 * traj;
}

R3 ArchimedeanSpiral(Info const &info)
{
  R3 traj(3, info.read_points, info.spokes_total());
  if (info.spokes_lo) {
    R3 lo = ASpiral(info.read_points, info.spokes_lo);
    traj.slice(Sz3{0, 0, 0}, Sz3{3, info.read_points, info.spokes_lo}) = lo;
  }
  R3 hi = ASpiral(info.read_points, info.spokes_hi);
  traj.slice(Sz3{0, 0, info.spokes_lo}, Sz3{3, info.read_points, info.spokes_hi}) = hi;
  return traj;
}

long Fib(long n)
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

R3 Phyllotaxis(Info const &info, long const smoothness, long const spi, bool const gm)
{
  long const nRead = info.read_points;
  long const nSpokes = info.spokes_total();
  if ((nSpokes % spi) != 0) {
    Log::Fail("Spokes per interleave {} is not a divisor of total spokes {}", spi, nSpokes);
  }
  long nInterleaves = nSpokes / spi;
  constexpr float phi_gold = 2.399963229728653;
  constexpr float phi_gm1 = 0.465571231876768;
  constexpr float phi_gm2 = 0.682327803828019;

  float dphi = phi_gold * Fib(smoothness);

  R1 read(nRead);
  for (long ir = 0; ir < nRead; ir++) {
    read(ir) = (float)(ir) / nRead;
  }
  // Currently to do an outer product, you need to contract over empty indices
  Eigen::array<Eigen::IndexPair<long>, 0> empty = {};

  R3 traj(3, nRead, nSpokes);
  R1 endPoint(3);
  for (long ii = 0; ii < nInterleaves; ii++) {
    float const z_ii = (ii * 2.) / (nSpokes - 1);
    float const phi_ii = (ii * phi_gold);

    float const gm_phi = 2 * M_PI * ii * phi_gm2;
    float const gm_theta = acos(fmod(ii * phi_gm1, 1.0));

    for (long is = 0; is < spi; is++) {
      float const z = (1 - (2. * nInterleaves * is) / (nSpokes - 1.)) - z_ii;
      float const theta = acos(z);
      float const phi = (is * dphi) + phi_ii;
      endPoint(0) = sin(theta) * cos(phi);
      endPoint(1) = sin(theta) * sin(phi);
      endPoint(2) = z;
      if (gm) {
        R2 rot(3, 3);
        rot(0, 0) = cos(gm_theta) * cos(gm_phi);
        rot(0, 1) = -sin(gm_phi);
        rot(0, 2) = cos(gm_phi) * sin(gm_theta);
        rot(1, 0) = cos(gm_theta) * sin(gm_phi);
        rot(1, 1) = cos(gm_phi);
        rot(1, 2) = sin(gm_theta) * sin(gm_phi);
        rot(2, 0) = -sin(gm_theta);
        rot(2, 1) = 0;
        rot(2, 2) = cos(gm_theta);
        endPoint =
          rot.contract(endPoint, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>()).eval();
      }
      traj.chip((ii * spi) + is, 2) = endPoint.contract(read, empty);
    }
  }
  traj = traj * traj.constant(0.5); // Scale to -0.5 to 0.5
  return traj;
}
