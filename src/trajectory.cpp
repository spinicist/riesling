#include "trajectory.h"

#include "tensorOps.h"

namespace rl {

Trajectory::Trajectory() {}

Trajectory::Trajectory(Info const &info, Re3 const &points)
  : info_{info}
  , points_{points}

{
  frames_ = I1(info_.traces);
  frames_.setZero();
  init();
}

Trajectory::Trajectory(Info const &info, Re3 const &points, I1 const &fr)
  : info_{info}
  , points_{points}
  , frames_{fr}

{
  init();
}

void Trajectory::init()
{
  if (info_.samples != points_.dimension(1)) {
    Log::Fail("Mismatch between info read points {} and trajectory points {}", info_.samples, points_.dimension(1));
  }
  if (info_.traces != points_.dimension(2)) {
    Log::Fail("Mismatch between info traces {} and trajectory traces {}", info_.traces, points_.dimension(2));
  }
  if (info_.traces != frames_.dimension(0)) {
    Log::Fail("Mismatch between info traces {} and frames array {}", info_.traces, frames_.dimension(0));
  }
  if (info_.frames < Maximum(frames_)) {
    Log::Fail("Maximum frame {} exceeds number of frames in header {}", Maximum(frames_), info_.frames);
  }

  if (info_.grid3D) {
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

  Log::Print(FMT_STRING("Created trajectory object with {} traces"), info_.traces);
}

Info const &Trajectory::info() const
{
  return info_;
}

Re3 const &Trajectory::points() const
{
  return points_;
}

I1 const &Trajectory::frames() const
{
  return frames_;
}

Re1 Trajectory::point(int16_t const read, int32_t const spoke) const
{
  assert(read < info_.samples);
  assert(spoke < info_.traces);

  Re1 const p = points_.chip(spoke, 2).chip(read, 1);
  return p;
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
    std::transform(
      info_.matrix.begin(), info_.matrix.end(), dsInfo.matrix.begin(), [dsamp](Index const i) { return (i / dsamp); });
    scale = static_cast<float>(info_.matrix[0]) / dsInfo.matrix[0];
    dsInfo.voxel_size = info_.voxel_size * scale;
    if (!dsInfo.grid3D) {
      dsInfo.matrix[2] = info_.matrix[2];
      dsInfo.voxel_size[2] = info_.voxel_size[2];
    }
  }
  Index const sz = info_.grid3D ? 3 : 2; // Need this for slicing below
  Index minRead = info_.samples, maxRead = 0;
  Re3 dsPoints(points_.dimensions());
  for (Index is = 0; is < info_.traces; is++) {
    for (Index ir = 0; ir < info_.samples; ir++) {
      Re1 p = points_.chip<2>(is).chip<1>(ir);
      p.slice(Sz1{0}, Sz1{sz}) *= p.slice(Sz1{0}, Sz1{sz}).constant(scale);
      if (Norm(p.slice(Sz1{0}, Sz1{sz})) <= 0.5f) {
        dsPoints.chip<2>(is).chip<1>(ir) = p;
        if (is >= lores) { // Ignore lo-res traces for this calculation
          minRead = std::min(minRead, ir);
          maxRead = std::max(maxRead, ir);
        }
      } else {
        dsPoints.chip<2>(is).chip<1>(ir).setConstant(std::numeric_limits<float>::quiet_NaN());
      }
    }
  }
  dsInfo.samples = 1 + maxRead - minRead;
  Log::Print(
    FMT_STRING("Downsampled by {}, new voxel-size {} matrix {}, read-points {}-{}{}"),
    scale,
    dsInfo.voxel_size.transpose(),
    dsInfo.matrix,
    minRead,
    maxRead,
    lores > 0 ? fmt::format(FMT_STRING(", ignoring {} lo-res traces"), lores) : "");
  dsPoints = Re3(dsPoints.slice(Sz3{0, minRead, 0}, Sz3{3, dsInfo.samples, dsInfo.traces}));
  return std::make_tuple(Trajectory(dsInfo, dsPoints, frames_), minRead);
}

} // namespace rl
