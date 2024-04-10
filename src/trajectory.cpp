#include "trajectory.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

Trajectory::Trajectory(Re3 const &points, Sz3 const matrix, Eigen::Array3f const voxel_size)
  : points_{points}
  , matrix_{matrix}
  , voxel_size_{voxel_size}
{
  init();
}

Trajectory::Trajectory(HD5::Reader &file, Eigen::Array3f const voxel_size)
{

  points_ = file.readTensor<Re3>(HD5::Keys::Trajectory);
  if (file.exists(HD5::Keys::Trajectory, "matrix")) {
    matrix_ = file.readAttribute<Sz3>(HD5::Keys::Trajectory, "matrix");
  } else {
    Re1 const maxCoords = points_.abs().maximum(Sz2{1, 2});
    for (Index ii = 0; ii < points_.dimension(0); ii++) {
      matrix_[ii] = maxCoords(ii) * 2;
    }
  }
  voxel_size_ = voxel_size;
  init();
}

void Trajectory::init()
{
  Index const nD = points_.dimension(0);
  if (nD < 1 || nD > 3) { Log::Fail("Trajectory has {} dimensions", nD); }

  Index discarded = 0;
  Re1   mat(nD);
  for (Index ii = 0; ii < nD; ii++) {
    mat(ii) = matrix_[ii] / 2.f;
  }
  for (Index is = 0; is < points_.dimension(2); is++) {
    for (Index ir = 0; ir < points_.dimension(1); ir++) {
      if (B0((points_.chip<2>(is).chip<1>(ir).abs() > mat).any())()) {
        points_.chip<2>(is).chip<1>(ir).setConstant(std::numeric_limits<float>::quiet_NaN());
        discarded++;
      }
    }
  }
  if (discarded > 0) {
    Index const total = points_.dimension(1) * points_.dimension(2);
    float const percent = (100.f * discarded) / total;
    Log::Warn("Discarded {} trajectory points ({}%) > 0.5", discarded, percent);
  }
  Log::Debug("{}D Trajectory size {},{}", nDims(), nSamples(), nTraces());
}

void Trajectory::write(HD5::Writer &file) const
{
  file.writeTensor(HD5::Keys::Trajectory, points_.dimensions(), points_.data(), HD5::Dims::Trajectory);
  file.writeAttribute(HD5::Keys::Trajectory, "matrix", matrix_);
}

auto Trajectory::nDims() const -> Index { return points_.dimension(0); }

auto Trajectory::nSamples() const -> Index { return points_.dimension(1); }

auto Trajectory::nTraces() const -> Index { return points_.dimension(2); }

void Trajectory::checkDims(Sz3 const dims) const
{
  if (dims[1] != nSamples()) { Log::Fail("Number of samples in data {} does not match trajectory {}", dims[1], nSamples()); }
  if (dims[2] != nTraces()) { Log::Fail("Number of traces in data {} does not match trajectory {}", dims[2], nTraces()); }
}

auto Trajectory::compatible(Trajectory const &other) const -> bool
{
  if ((other.matrix() == matrix()) && (other.voxelSize() == voxelSize()).all() && (other.nDims() == nDims())) {
    return true;
  } else {
    return false;
  }
}

auto Trajectory::matrix() const -> Sz3 { return matrix_; }

auto Trajectory::voxelSize() const -> Eigen::Array3f { return voxel_size_; }

auto Trajectory::FOV() const -> Eigen::Array3f
{
  Eigen::Array3f fov;
  for (Index ii = 0; ii < 3; ii++) {
    fov[ii] = matrix_[ii] * voxel_size_[ii];
  }
  return fov;
}

auto Trajectory::matrixForFOV(float const fov) const -> Sz3
{
  if (fov > 0) {
    Eigen::Array3l bigMatrix = (((fov / voxel_size_) / 2.f).floor() * 2).cast<Index>();
    Sz3            matrix = matrix_;
    for (Index ii = 0; ii < 3; ii++) {
      matrix[ii] = bigMatrix[ii];
    }
    Log::Print("Requested FOV {} from matrix {}, calculated {}", fov, matrix_, matrix);
    return matrix;
  } else {
    return matrix_;
  }
}

auto Trajectory::matrixForFOV(Eigen::Array3f const fov) const -> Sz3
{
  Sz3 matrix;
  for (Index ii = 0; ii < 3; ii++) {
    matrix[ii] = std::max(matrix_[ii], 2 * (Index)(fov[ii] / voxel_size_[ii] / 2.f));
  }
  Log::Print("Requested FOV {} from matrix {}, calculated {}", fov.transpose(), matrix_, matrix);
  return matrix;
}

Re3 const &Trajectory::points() const { return points_; }

Re1 Trajectory::point(int16_t const read, int32_t const spoke) const
{
  Re1 const p = points_.chip<2>(spoke).chip<1>(read);
  return p;
}

auto Trajectory::downsample(Eigen::Array3f const tgtSize, Index const lores, bool const shrink, bool const corners) const
  -> std::tuple<Trajectory, Index, Index>
{
  Eigen::Array3f ratios = voxel_size_ / tgtSize;
  if ((ratios > 1.f).any()) {
    Log::Fail("Downsample voxel-size {} is larger than current voxel-size {}", tgtSize, voxel_size_);
  }
  auto  dsVox = voxel_size_;
  auto  dsMatrix = matrix_;
  Re1   thresh(nDims());
  for (Index ii = 0; ii < nDims(); ii++) {
    if (shrink) {
      // Account for rounding
      dsMatrix[ii] = matrix_[ii] * ratios[ii];
      float const scale = static_cast<float>(matrix_[0]) / dsMatrix[0];
      ratios(ii) = 1.f / scale;
      dsVox[ii] = voxel_size_[ii] * scale;
    }
    thresh(ii) = matrix_[ii] * ratios(ii) / 2.f;
  }
  Index minSamp = nSamples(), maxSamp = 0;
  Re3   dsPoints(points_.dimensions());
  for (Index it = 0; it < nTraces(); it++) {
    for (Index is = 0; is < nSamples(); is++) {
      Re1 p = points_.chip<2>(it).chip<1>(is);
      if ((corners && B0((p.abs() <= thresh).all())()) || Norm(p / thresh) <= 1.f) {
        dsPoints.chip<2>(it).chip<1>(is) = p;
        for (int ii = 0; ii < nDims(); ii++) {
          p(ii) /= ratios(ii);
        }
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
  Log::Print("Target res {} mm, ratios {}, matrix {}, voxel-size {} mm, read-points {}-{}{}", tgtSize,
             fmt::streamed(ratios.transpose()), dsMatrix, dsVox.transpose(), minSamp, maxSamp,
             lores > 0 ? fmt::format(", ignoring {} lo-res traces", lores) : "");
  dsPoints = Re3(dsPoints.slice(Sz3{0, minSamp, 0}, Sz3{nDims(), dsSamples, nTraces()}));
  Log::Print("Downsampled trajectory dims {}", dsPoints.dimensions());
  return std::make_tuple(Trajectory(dsPoints, dsMatrix, dsVox), minSamp, dsSamples);
}

auto Trajectory::downsample(Cx5 const           &ks,
                            Eigen::Array3f const tgt,
                            Index const          lores,
                            bool const           shrink,
                            bool const           corners) const -> std::tuple<Trajectory, Cx5>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(tgt, lores, shrink, corners);
  Cx5 dsKs = ks.slice(Sz5{0, minSamp, 0, 0, 0}, Sz5{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3), ks.dimension(4)});
  return std::make_tuple(dsTraj, dsKs);
}

auto Trajectory::downsample(Cx4 const           &ks,
                            Eigen::Array3f const tgt,
                            Index const          lores,
                            bool const           shrink,
                            bool const           corners) const -> std::tuple<Trajectory, Cx4>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(tgt, lores, shrink, corners);
  Cx4 dsKs = ks.slice(Sz4{0, minSamp, 0, 0}, Sz4{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3)});
  return std::make_tuple(dsTraj, dsKs);
}

} // namespace rl
