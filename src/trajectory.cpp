#include "trajectory.h"

#include "tensorOps.h"

Trajectory::Trajectory() {}

Trajectory::Trajectory(Info const &info, R3 const &points)
  : info_{info}
  , points_{points}

{
  frames_ = I1(info_.spokes);
  frames_.setZero();
  init();
}

Trajectory::Trajectory(Info const &info, R3 const &points, I1 const &fr)
  : info_{info}
  , points_{points}
  , frames_{fr}

{
  init();
}

void Trajectory::init()
{
  if (info_.read_points != points_.dimension(1)) {
    Log::Fail("Mismatch between info read points {} and trajectory points {}", info_.read_points, points_.dimension(1));
  }
  if (info_.spokes != points_.dimension(2)) {
    Log::Fail("Mismatch between info spokes {} and trajectory spokes {}", info_.spokes, points_.dimension(2));
  }
  if (info_.spokes != frames_.dimension(0)) {
    Log::Fail("Mismatch between info spokes {} and frames array {}", info_.spokes, frames_.dimension(0));
  }
  if (info_.frames < Maximum(frames_)) {
    Log::Fail("Maximum frame {} exceeds number of frames in header {}", Maximum(frames_), info_.frames);
  }

  if (info_.type == Info::Type::ThreeD) {
    float const maxCoord = Maximum(points_.abs());
    if (maxCoord > 0.5f) {
      Log::Fail(FMT_STRING("Maximum trajectory co-ordinate {} exceeded 0.5"), maxCoord);
    }
  } else {
    float const maxCoord =
      Maximum(points_.slice(Sz3{0, 0, 0}, Sz3{2, points_.dimension(1), points_.dimension(2)}).abs());
    if (maxCoord > 0.5f) {
      Log::Fail(FMT_STRING("Maximum in-plane trajectory {} co-ordinate exceeded 0.5"), maxCoord);
    }
  }

  Log::Print(FMT_STRING("Created trajectory object with {} spokes"), info_.spokes);
}

Info const &Trajectory::info() const
{
  return info_;
}

R3 const &Trajectory::points() const
{
  return points_;
}

I1 const &Trajectory::frames() const
{
  return frames_;
}

Point3 Trajectory::point(int16_t const read, int32_t const spoke, float const rad_hi) const
{
  assert(read < info_.read_points);
  assert(spoke < info_.spokes);

  // Convention is to store the points between -0.5 and 0.5, so we need a factor of 2 here
  float const diameter = 2.f * rad_hi;
  R1 const p = points_.chip(spoke, 2).chip(read, 1);
  switch (info_.type) {
  case Info::Type::ThreeD:
    return Point3{p(0) * diameter, p(1) * diameter, p(2) * diameter};
  case Info::Type::ThreeDStack:
    return Point3{p(0) * diameter, p(1) * diameter, p(2) - (info_.matrix[2] / 2)};
  }
  __builtin_unreachable(); // Because the GCC devs are very obtuse
}

std::tuple<Trajectory, Index> Trajectory::downsample(float const res, Index const lores, bool const shrink) const
{
  float const dsamp = res / info_.voxel_size.minCoeff();
  if (dsamp < 1.f) {
    Log::Fail(
      FMT_STRING("Downsample resolution {} is lower than input resolution {}"), res, info_.voxel_size.minCoeff());
  }
  auto dsInfo = info_;
  float scale = 1.f;
  if (shrink) {
    // Account for rounding
    dsInfo.matrix = (info_.matrix.cast<float>() / dsamp).cast<Index>();
    scale = static_cast<float>(info_.matrix[0]) / dsInfo.matrix[0];
    dsInfo.voxel_size = info_.voxel_size * scale;
    if (dsInfo.type == Info::Type::ThreeDStack) {
      dsInfo.matrix[2] = info_.matrix[2];
      dsInfo.voxel_size[2] = info_.voxel_size[2];
    }
  }
  Index const sz = (info_.type == Info::Type::ThreeD) ? 3 : 2; // Need this for slicing below
  Index minRead = info_.read_points, maxRead = 0;
  R3 dsPoints(points_.dimensions());
  for (Index is = 0; is < info_.spokes; is++) {
    for (Index ir = 0; ir < info_.read_points; ir++) {
      R1 p = points_.chip<2>(is).chip<1>(ir);
      p.slice(Sz1{0}, Sz1{sz}) *= p.slice(Sz1{0}, Sz1{sz}).constant(scale);
      if (Norm(p.slice(Sz1{0}, Sz1{sz})) <= 0.5f) {
        dsPoints.chip<2>(is).chip<1>(ir) = p;
        if (is >= lores) { // Ignore lo-res spokes for this calculation
          minRead = std::min(minRead, ir);
          maxRead = std::max(maxRead, ir);
        }
      } else {
        dsPoints.chip<2>(is).chip<1>(ir).setConstant(std::numeric_limits<float>::quiet_NaN());
      }
    }
  }
  dsInfo.read_points = 1 + maxRead - minRead;
  Log::Print(
    FMT_STRING("Downsampled by {}, new voxel-size {} matrix {}, read-points {}-{}{}"),
    scale,
    dsInfo.voxel_size.transpose(),
    dsInfo.matrix.transpose(),
    minRead,
    maxRead,
    lores > 0 ? fmt::format(FMT_STRING(", ignoring {} lo-res spokes"), lores) : "");
  dsPoints = R3(dsPoints.slice(Sz3{0, minRead, 0}, Sz3{3, dsInfo.read_points, dsInfo.spokes}));
  return std::make_tuple(Trajectory(dsInfo, dsPoints, frames_), minRead);
}