#include "traj_archimedean.h"

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

  return traj;
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