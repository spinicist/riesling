#include "radial.hpp"
#include "../log/log.hpp"
#include "../tensors.hpp"

namespace rl {

namespace {
Re2 Spoke(Index const nRead)
{
  Re2 points(3, nRead);
  points.setZero();
  for (Index ir = 0; ir < nRead; ir++) {
    points(2, ir) = (float)(ir) / (nRead - 1);
  }
  return points;
}

constexpr Eigen::IndexPairList<Eigen::type2indexpair<1, 0>> MatMul;

Re2 Rotation(float const phi, float const theta)
{
  Re2 rot(3, 3);
  rot(0, 0) = sin(theta) * cos(phi);
  rot(0, 1) = -sin(phi);
  rot(0, 2) = cos(phi) * cos(theta);
  rot(1, 0) = sin(theta) * sin(phi);
  rot(1, 1) = cos(phi);
  rot(1, 2) = cos(theta) * sin(phi);
  rot(2, 0) = -cos(theta);
  rot(2, 1) = 0;
  rot(2, 2) = sin(theta);
  return rot;
}

} // namespace

/* This implements the Archimedean spiral described in S. T. S. Wong and M. S. Roos, ‘A strategy for
 * sampling on a sphere applied to 3D selective RF pulse design’, Magnetic Resonance in Medicine,
 * vol. 32, no. 6, pp. 778–784, Dec. 1994, doi: 10.1002/mrm.1910320614.
 */
Re3 ArchimedeanSpiral(Index const matrix, float const OS, Index const nSpoke)
{
  Index const nRead = OS * matrix / 2.f;
  Re3         traj(3, nRead, nSpoke);
  Re2 const   read = Spoke(nRead);

  // Slightly annoying as can't initialize a 1D Tensor with {} notation yet
  traj.chip(0, 2) = Rotation(0.f, M_PI / 2.f).contract(read, MatMul);
  Index const half = nSpoke / 2;
  traj.chip(half, 2) = Rotation(0.f, 0.f).contract(read, MatMul);

  double const dz = 1. / half;          // Change in theta
  double const c2 = nSpoke * 4. * M_PI; // The velocity squared
  double       z = 0.;
  double       phi = 0.;
  for (Index is = 1; is < half; is++) {
    z += dz;
    double const cos_z2 = z * z;
    double const sin_z2 = 1. - cos_z2;
    double const d_phi = 0.5 * dz * std::sqrt((1. / sin_z2) * (c2 - (1. / sin_z2)));
    phi += d_phi;
    double const t = std::asin(z);
    traj.chip(half - is, 2) = Rotation(phi, t).contract(read, MatMul);
    traj.chip(half + is, 2) = Rotation(-phi, -t).contract(read, MatMul);
  }
  return traj * traj.constant(matrix / 2.f);
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

Re3 Phyllotaxis(Index const matrix, float const OS, Index const ntraces, Index const smoothness, Index const spi)
{
  if ((ntraces % spi) != 0) {
    throw Log::Failure("Phyll", "Traces per interleave {} is not a divisor of total traces {}", spi, ntraces);
  }
  Index           nInterleaves = ntraces / spi;
  constexpr float phi_gold = 2.399963229728653;
  float           dphi = phi_gold * Fib(smoothness);
  Index const     nRead = OS * matrix / 2.f;
  Re2 const       read = Spoke(nRead);

  Re3 traj(3, nRead, ntraces);
  Re1 endPoint(3);
  for (Index ii = 0; ii < nInterleaves; ii++) {
    float const z_ii = (ii * 2.) / (ntraces - 1);
    float const phi_ii = (ii * phi_gold);
    for (Index is = 0; is < spi; is++) {
      float const z = (1 - (2. * nInterleaves * is) / (ntraces - 1.)) - z_ii;
      float const theta = std::asin(z);
      float const phi = (is * dphi) + phi_ii;
      traj.chip((ii * spi) + is, 2) = Rotation(phi, theta).contract(read, MatMul);
    }
  }
  traj = traj * traj.constant(matrix / 2.f);
  return traj;
}
} // namespace rl
