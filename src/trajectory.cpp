#include "trajectory.hpp"

#include "tensorOps.hpp"

namespace rl {

Trajectory::Trajectory() {}

Trajectory::Trajectory(Info const &info, Re3 const &points)
  : info_{info}
  , points_{points}

{
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
  if (frames_.size() && nTraces() != frames_.dimension(0)) {
    Log::Fail("Mismatch between number of traces {} and frames array {}", nTraces(), frames_.dimension(0));
  }
  float const maxCoord = Maximum(points_.abs());
  if (maxCoord > 0.5f) {
    Log::Fail(FMT_STRING("Maximum trajectory co-ordinate {} exceeded 0.5"), maxCoord);
  }
  Log::Print<Log::Level::Debug>(FMT_STRING("Trajectory size {},{},{}"), nSamples(), nTraces(), nFrames());
}

auto Trajectory::nSamples() const -> Index
{
  return points_.dimension(1);
}

auto Trajectory::nTraces() const -> Index
{
  return points_.dimension(2);
}

auto Trajectory::nFrames() const -> Index
{
  if (frames_.size()) {
    return Maximum(frames_) + 1;
  } else {
    return 1;
  }
}

Info const &Trajectory::info() const
{
  return info_;
}

Re3 const &Trajectory::points() const
{
  return points_;
}

Index Trajectory::frame(Index const i) const {
  if (frames_.size()) {
    return frames_(i);
  } else {
    return 0;
  }
}

I1 const &Trajectory::frames() const
{
  return frames_;
}

Re1 Trajectory::point(int16_t const read, int32_t const spoke) const
{
  Re1 const p = points_.chip(spoke, 2).chip(read, 1);
  return p;
}

auto Trajectory::downsample(float const res, Index const lores, bool const shrink) const -> std::tuple<Trajectory, Index, Index>
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
  }
  Index minSamp = nSamples(), maxSamp = 0;
  Re3 dsPoints(points_.dimensions());
  for (Index it = 0; it < nTraces(); it++) {
    for (Index is = 0; is < nSamples(); is++) {
      Re1 p = points_.chip<2>(it).chip<1>(is);
      p *= p.constant(scale);
      if (Norm(p) <= 0.5f) {
        dsPoints.chip<2>(it).chip<1>(is) = p;
      } else {
        if (it >= lores) { // Ignore lo-res traces for this calculation
          minSamp = std::min(minSamp, is);
          maxSamp = std::max(maxSamp, is);
        }
        dsPoints.chip<2>(it).chip<1>(is).setConstant(std::numeric_limits<float>::quiet_NaN());
      }
    }
  }
  Index const dsSamples = maxSamp + 1 - minSamp;
  Log::Print(
    FMT_STRING("Downsampled by {}, new voxel-size {} matrix {}, read-points {}-{}{}"),
    scale,
    dsInfo.voxel_size.transpose(),
    dsInfo.matrix,
    minSamp,
    maxSamp,
    lores > 0 ? fmt::format(FMT_STRING(", ignoring {} lo-res traces"), lores) : "");
  dsPoints = Re3(dsPoints.slice(Sz3{0, minSamp, 0}, Sz3{3, dsSamples, nTraces()}));
  return std::make_tuple(Trajectory(dsInfo, dsPoints, frames_), minSamp, dsSamples);
}

auto Trajectory::downsample(Cx4 const &ks, float const res, Index const lores, bool const shrink) const
  -> std::tuple<Trajectory, Cx4>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(res, lores, shrink);
  Cx4 dsKs = ks.slice(Sz4{0, minSamp, 0, 0}, Sz4{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3)});
  return std::make_tuple(dsTraj, dsKs);
}

auto Trajectory::downsample(Cx3 const &ks, float const res, Index const lores, bool const shrink) const
  -> std::tuple<Trajectory, Cx3>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(res, lores, shrink);
  Cx3 dsKs = ks.slice(Sz3{0, minSamp, 0}, Sz3{ks.dimension(0), nSamp, ks.dimension(2)});
  return std::make_tuple(dsTraj, dsKs);
}

} // namespace rl
