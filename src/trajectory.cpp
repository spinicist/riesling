#include "trajectory.hpp"

#include "tensorOps.hpp"

namespace rl {

Trajectory::Trajectory() {}

Trajectory::Trajectory(Info const &info, Re3 const &points)
  : info_{info}
  , points_{points}
{
  if (points_.dimension(0) < 1 || points_.dimension(0) > 3) { Log::Fail("Trajectory has {} dimensions", points_.dimension(0)); }

  float const maxCoord = Maximum(points_.abs());
  if (maxCoord > 0.5f) { Log::Warn("Maximum trajectory co-ordinate {} > 0.5", maxCoord); }
  Log::Print<Log::Level::Debug>("{}D Trajectory size {},{}", nDims(), nSamples(), nTraces());
}

auto Trajectory::nDims() const -> Index { return points_.dimension(0); }

auto Trajectory::nSamples() const -> Index { return points_.dimension(1); }

auto Trajectory::nTraces() const -> Index { return points_.dimension(2); }

void Trajectory::checkDims(Sz3 const dims) const
{
  if (dims[1] != nSamples()) { Log::Fail("Number of samples in data {} does not match trajectory {}", dims[1], nSamples()); }
  if (dims[2] != nTraces()) { Log::Fail("Number of traces in data {} does not match trajectory {}", dims[2], nTraces()); }
}

Info const &Trajectory::info() const { return info_; }

auto Trajectory::matrix() const -> Sz3 { return info_.matrix; }

auto Trajectory::matrixForFOV(float const fov) const -> Sz3
{
  if (fov > 0) {
    Eigen::Array3l bigMatrix = (((fov / info_.voxel_size) / 2.f).floor() * 2).cast<Index>();
    Sz3            matrix = info_.matrix;
    for (Index ii = 0; ii < nDims(); ii++) {
      matrix[ii] = bigMatrix[ii];
    }
    Log::Print<Log::Level::High>("Requested FOV {} from matrix {}, calculated {}", fov, info_.matrix, matrix);
    return matrix;
  } else {
    return info_.matrix;
  }
}

auto Trajectory::matrixForFOV(Eigen::Array3f const fov) const -> Sz3
{
  Sz3 matrix;
  for (Index ii = 0; ii < nDims(); ii++) {
      matrix[ii] = std::max(info_.matrix[ii], 2 * (Index)(fov[ii] / info_.voxel_size[ii] / 2.f));
  }
  Log::Print<Log::Level::High>("Requested FOV {} from matrix {}, calculated {}", fov.transpose(), info_.matrix, matrix);
  return matrix;
}

Re3 const &Trajectory::points() const { return points_; }

Re1 Trajectory::point(int16_t const read, int32_t const spoke) const
{
  Re1 const p = points_.chip<2>(spoke).chip<1>(read);
  return p;
}

auto Trajectory::downsample(float const res, Index const lores, bool const shrink, bool const corners) const
  -> std::tuple<Trajectory, Index, Index>
{
  float dsamp = info_.voxel_size.minCoeff() / res;
  if (dsamp > 1.f) {
    Log::Fail("Downsample resolution {} is higher than input resolution {}", res, info_.voxel_size.minCoeff());
  }
  auto  dsInfo = info_;
  float scale = 1.f;
  if (shrink) {
    // Account for rounding
    std::transform(info_.matrix.begin(), info_.matrix.begin() + nDims(), dsInfo.matrix.begin(),
                   [dsamp](Index const i) { return (i * dsamp); });
    scale = static_cast<float>(info_.matrix[0]) / dsInfo.matrix[0];
    dsamp = 1.f / scale;
    dsInfo.voxel_size = info_.voxel_size * scale;
  }
  Index       minSamp = nSamples(), maxSamp = 0;
  Re3         dsPoints(points_.dimensions());
  float const thresh = 0.5f * dsamp;
  for (Index it = 0; it < nTraces(); it++) {
    for (Index is = 0; is < nSamples(); is++) {
      Re1 p = points_.chip<2>(it).chip<1>(is);
      if ((corners && B0((p.abs() <= thresh).all())()) || Norm(p) <= thresh) {
        dsPoints.chip<2>(it).chip<1>(is) = p * scale;
        if (it >= lores) { // Ignore lo-res traces for this calculation
          minSamp = std::min(minSamp, is);
          maxSamp = std::max(maxSamp, is);
        }
      } else {
        dsPoints.chip<2>(it).chip<1>(is).setConstant(std::numeric_limits<float>::quiet_NaN());
      }
    }
  }
  if (minSamp > maxSamp) { Log::Fail("No valid trajectory points remain after downsampling"); }
  Index const dsSamples = maxSamp + 1 - minSamp;
  Log::Print("Target res {} mm, factor {}, matrix {}, voxel-size {} mm, read-points {}-{}{}", res, dsamp, dsInfo.matrix,
             dsInfo.voxel_size.transpose(), minSamp, maxSamp,
             lores > 0 ? fmt::format(", ignoring {} lo-res traces", lores) : "");
  dsPoints = Re3(dsPoints.slice(Sz3{0, minSamp, 0}, Sz3{nDims(), dsSamples, nTraces()}));
  Log::Print("Downsampled trajectory dims {}", dsPoints.dimensions());
  return std::make_tuple(Trajectory(dsInfo, dsPoints), minSamp, dsSamples);
}

auto Trajectory::downsample(Cx5 const &ks, float const res, Index const lores, bool const shrink, bool const corners) const
  -> std::tuple<Trajectory, Cx5>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(res, lores, shrink, corners);
  Cx5 dsKs = ks.slice(Sz5{0, minSamp, 0, 0, 0}, Sz5{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3), ks.dimension(4)});
  return std::make_tuple(dsTraj, dsKs);
}

auto Trajectory::downsample(Cx4 const &ks, float const res, Index const lores, bool const shrink, bool const corners) const
  -> std::tuple<Trajectory, Cx4>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(res, lores, shrink, corners);
  Cx4 dsKs = ks.slice(Sz4{0, minSamp, 0, 0}, Sz4{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3)});
  return std::make_tuple(dsTraj, dsKs);
}

} // namespace rl
