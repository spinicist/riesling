#include "trajectory.h"

#include "tensorOps.h"

Trajectory::Trajectory(Info const &info, R3 const &points, Log const &log)
    : info_{info}
    , points_{points}
    , log_{log}
{
  if (info_.read_points != points_.dimension(1)) {
    log_.fail(
        "Mismatch between info read points {} and trajectory points {}",
        info_.read_points,
        points_.dimension(1));
  }
  if (info_.spokes_total() != points_.dimension(2)) {
    log_.fail(
        "Mismatch between info spokes {} and trajectory spokes {}",
        info_.spokes_total(),
        points_.dimension(2));
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

  float const rad = (spoke < info_.spokes_lo) ? rad_hi / info_.lo_scale : rad_hi;
  R1 const p = points_.chip(spoke, 2).chip(read, 1);
  switch (info_.type) {
  case Info::Type::ThreeD:
    return Point3{p(0) * rad, p(1) * rad, p(2) * rad};
  case Info::Type::ThreeDStack:
    return Point3{p(0) * rad, p(1) * rad, p(2)};
  }
}

float Trajectory::merge(int16_t const read, int32_t const spoke) const
{
  if (spoke < info_.spokes_lo) {
    return mergeLo_(read);
  } else {
    return mergeHi_(read);
  }
}

Trajectory Trajectory::trim(float const res, Cx3 &data) const
{
  if (res <= 0.f) {
    log_.fail("Asked for trajectory with resolution {} which is less than or equal to zero", res);
  }

  float const ratio = info_.voxel_size.minCoeff() / res;
  log_.info("Image res {} SENSE res {} ratio {}", info_.voxel_size.minCoeff(), res, ratio);
  // Assume radial spokes for now
  Info new_info = info_;

  int16_t lo = std::numeric_limits<int16_t>::max();
  int16_t hi = std::numeric_limits<int16_t>::min();

  for (int16_t ir = info_.read_gap; ir < info_.read_points; ir++) {
    float const rad = point(ir, info_.spokes_lo, 1.f).norm();
    if (rad <= ratio) { // Discard points above the desired resolution
      lo = std::min(ir, lo);
      hi = std::max(ir, hi);
    }
  }
  new_info.read_points = hi - lo;
  log_.info("Trimming data to read points {}-{}", lo, hi);
  R3 new_points = points_.slice(Sz3{0, lo, 0}, Sz3{3, new_info.read_points, info_.spokes_total()});
  Cx3 const temp =
      data.slice(Sz3{0, lo, 0}, Sz3{data.dimension(0), new_info.read_points, data.dimension(2)});
  data = temp;
  return Trajectory(new_info, new_points, log_);
}
