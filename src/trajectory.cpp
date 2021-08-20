#include "trajectory.h"

#include "tensorOps.h"

Trajectory::Trajectory(Info const &info, R3 const &points, Log const &log)
    : info_{info}
    , points_{points}
    , log_{log}
{
  if (info_.read_points != points_.dimension(1)) {
    Log::Fail(
        "Mismatch between info read points {} and trajectory points {}",
        info_.read_points,
        points_.dimension(1));
  }
  if (info_.spokes_total() != points_.dimension(2)) {
    Log::Fail(
        "Mismatch between info spokes {} and trajectory spokes {}",
        info_.spokes_total(),
        points_.dimension(2));
  }

  if (info_.type == Info::Type::ThreeD) {
    float const maxCoord = Maximum(points_.abs());
    if (maxCoord > 0.5f) {
      log_.fail("Maximum trajectory co-ordinate {} exceeded 0.5", maxCoord);
    }
  } else {
    float const maxCoord = Maximum(
        points_.slice(Sz3{0, 0, 0}, Sz3{2, points_.dimension(1), points_.dimension(2)}).abs());
    if (maxCoord > 0.5f) {
      log_.fail("Maximum in-plane trajectory {} co-ordinate exceeded 0.5", maxCoord);
    }
  }

  Eigen::ArrayXf ind = Eigen::ArrayXf::LinSpaced(info_.read_points, 0, info_.read_points - 1);
  mergeHi_ = ind - (info_.read_gap - 1);
  mergeHi_ = (mergeHi_ > 0).select(mergeHi_, 0);
  mergeHi_ = (mergeHi_ < 1).select(mergeHi_, 1);

  if (info_.spokes_lo) {
    ind = Eigen::ArrayXf::LinSpaced(info_.read_points, 0, info_.read_points - 1);
    mergeLo_ = ind / info_.lo_scale - (info_.read_gap - 1);
    mergeLo_ = (mergeLo_ > 0).select(mergeLo_, 0);
    mergeLo_ = (mergeLo_ < 1).select(mergeLo_, 1);
    mergeLo_ = (1 - mergeLo_) / info_.lo_scale; // Match intensities of k-space
    mergeLo_.head(info_.read_gap) = 0.;         // Don't touch these points
  }
  log_.info("Created trajectory object with {} spokes", info_.spokes_total());
}

Info const &Trajectory::info() const
{
  return info_;
}

R3 const &Trajectory::points() const
{
  return points_;
}

Point3 Trajectory::point(int16_t const read, int32_t const spoke, float const rad_hi) const
{
  assert(read < info_.read_points);
  assert(spoke < info_.spokes_total());

  // Convention is to store the points between -0.5 and 0.5, so we need a factor of 2 here
  float const diameter = 2.f * (spoke < info_.spokes_lo ? rad_hi / info_.lo_scale : rad_hi);
  R1 const p = points_.chip(spoke, 2).chip(read, 1);
  switch (info_.type) {
  case Info::Type::ThreeD:
    return Point3{p(0) * diameter, p(1) * diameter, p(2) * diameter};
  case Info::Type::ThreeDStack:
    return Point3{p(0) * diameter, p(1) * diameter, p(2)};
  }
  __builtin_unreachable(); // Because the GCC devs are very obtuse
}

float Trajectory::merge(int16_t const read, int32_t const spoke) const
{
  if (spoke < info_.spokes_lo) {
    return mergeLo_(read);
  } else {
    return mergeHi_(read);
  }
}

Trajectory Trajectory::trim(float const res, Cx3 &data, bool const shrink) const
{
  if (res <= 0.f) {
    Log::Fail("Asked for trajectory with resolution {} which is less than or equal to zero", res);
  }

  float const ratio = info_.voxel_size.minCoeff() / res;
  log_.info(FMT_STRING("Cropping data to {} mm effective resolution, ratio {}"), res, ratio);
  // Assume radial spokes for now
  Info new_info = info_;

  int16_t hi = std::numeric_limits<int16_t>::min();
  for (int16_t ir = info_.read_gap; ir < info_.read_points; ir++) {
    float const rad = point(ir, info_.spokes_lo, 1.f).norm();
    if (rad <= ratio) { // Discard points above the desired resolution
      hi = std::max(ir, hi);
    }
  }
  new_info.read_points = hi;
  log_.info("Trimming data to read points {}-{}", 0, hi);
  R3 new_points = points_.slice(Sz3{0, 0, 0}, Sz3{3, new_info.read_points, info_.spokes_total()});
  if (shrink) {
    new_info.matrix = (info_.matrix.cast<float>() * ratio).cast<long>();
    // Assume this is the maximum radius
    new_points = new_points / ratio;
    log_.info(
        FMT_STRING("Reducing matrix from {} to {}"),
        info_.matrix.transpose(),
        new_info.matrix.transpose());
  }

  Cx3 const temp =
      data.slice(Sz3{0, 0, 0}, Sz3{data.dimension(0), new_info.read_points, data.dimension(2)});
  data = temp;
  return Trajectory(new_info, new_points, log_);
}
