#include "trajectory.hpp"

#include "log.hpp"
#include "tensorOps.hpp"

namespace rl {

/* Temp Hack because .maximum() may be buggy on NEON */
auto GuessMatrix(Re3 const &points) -> Sz3
{
  Re1 max(3);
  max.setZero();
  for (Index ii = 0; ii < points.dimension(1); ii++) {
    for (Index ij = 0; ij < points.dimension(2); ij++) {
      for (Index ic = 0; ic < 3; ic++) {
        auto const a = std::fabs(points(ic, ii, ij));
        if (a > max(ic)) { max(ic) = a; }
      }
    }
  }
  Sz3 mat;
  for (Index ii = 0; ii < 3; ii++) {
    mat[ii] = std::max((Index)(std::ceil(max(ii)) * 2), 1L);
  }
  return mat;
}

Trajectory::Trajectory(Re3 const &points, Eigen::Array3f const voxel_size)
  : points_{points}
  , matrix_{GuessMatrix(points_)}
  , voxel_size_{voxel_size}
{
  init();
}

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
    matrix_ = GuessMatrix(points_);
  }
  voxel_size_ = voxel_size;
  init();
}

void Trajectory::init()
{
  Index const nD = points_.dimension(0);
  if (nD != 3) { Log::Fail("Trajectory data has {}D co-ordinates, must be 3", nD); }

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
    Log::Warn("Discarded {} trajectory points ({:.2f}%) outside matrix", discarded, percent);
  }
  Log::Print("Trajectory: samples {} traces {} matrix {}", nSamples(), nTraces(), matrix_);
}

void Trajectory::write(HD5::Writer &file) const
{
  file.writeTensor(HD5::Keys::Trajectory, points_.dimensions(), points_.data(), HD5::Dims::Trajectory);
  file.writeAttribute(HD5::Keys::Trajectory, "matrix", matrix_);
}

auto Trajectory::nSamples() const -> Index { return points_.dimension(1); }

auto Trajectory::nTraces() const -> Index { return points_.dimension(2); }

void Trajectory::checkDims(Sz3 const dims) const
{
  if (dims[1] != nSamples()) { Log::Fail("Number of samples in data {} does not match trajectory {}", dims[1], nSamples()); }
  if (dims[2] != nTraces()) { Log::Fail("Number of traces in data {} does not match trajectory {}", dims[2], nTraces()); }
}

auto Trajectory::compatible(Trajectory const &other) const -> bool
{
  if ((other.matrix() == matrix()) && (other.voxelSize() == voxelSize()).all()) {
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

void Trajectory::shiftFOV(Eigen::Vector3f const shift, Cx5 &data)
{
  Re1 delta(3);
  delta[0] = shift[0] / (voxel_size_[0] * matrix_[0]);
  delta[1] = shift[1] / (voxel_size_[1] * matrix_[1]);
  delta[2] = shift[2] / (voxel_size_[2] * matrix_[2]);

  Eigen::IndexPairList<Eigen::type2indexpair<0, 0>> zero2zero;
  Log::Print("Shifting FOV by {} {} {}", delta[0], delta[1], delta[2]);

  auto const shape = data.dimensions();

  // Check for NaNs (trajectory points that should be ignored) and zero the corresponding data points. Otherwise they become
  // NaN, and cause problems in iterative recons
  data.device(Threads::GlobalDevice()) = data * points_.sum(Sz1{0})
                                                  .isfinite()
                                                  .reshape(Sz5{1, shape[1], shape[2], 1, 1})
                                                  .broadcast(Sz5{shape[0], 1, 1, 1, shape[4]})
                                                  .select((points_.contract(delta, zero2zero).cast<Cx>() * Cx(0.f, 2.f * M_PI))
                                                            .exp()
                                                            .reshape(Sz4{1, shape[1], shape[2], shape[3]})
                                                            .broadcast(Sz4{shape[0], 1, 1, 1})
                                                            .reshape(Sz5{shape[0], shape[1], shape[2], 1, shape[3]}),
                                                          data.constant(0.f));
}

Re3 const &Trajectory::points() const { return points_; }

Re1 Trajectory::point(int16_t const read, int32_t const spoke) const
{
  Re1 const p = points_.chip<2>(spoke).chip<1>(read);
  return p;
}

auto Trajectory::downsample(Eigen::Array3f const tgtSize,
                            Index const          fullResTraces,
                            bool const           shrink,
                            bool const           corners) const -> std::tuple<Trajectory, Index, Index>
{
  Eigen::Array3f ratios = voxel_size_ / tgtSize;
  if ((ratios > 1.f).any()) {
    Log::Fail("Downsample voxel-size {} is larger than current voxel-size {}", tgtSize, voxel_size_);
  }
  auto dsVox = voxel_size_;
  auto dsMatrix = matrix_;
  Re1  thresh(3);
  for (Index ii = 0; ii < 3; ii++) {
    if (shrink) {
      // Account for rounding
      dsMatrix[ii] = matrix_[ii] * ratios[ii];
      float const scale = static_cast<float>(matrix_[0]) / dsMatrix[0];
      ratios(ii) = 1.f / scale;
      dsVox[ii] = voxel_size_[ii] * scale;
    }
    thresh(ii) = matrix_[ii] * ratios(ii) / 2.f;
  }
  Log::Print("Downsampling to {} mm from {} mm, matrix {}, ratios {}", tgtSize, dsVox.transpose(), dsMatrix,
             fmt::streamed(ratios.transpose()));

  Index minSamp = nSamples(), maxSamp = 0;
  Re3   dsPoints(points_.dimensions());
  for (Index it = 0; it < nTraces(); it++) {
    for (Index is = 0; is < nSamples(); is++) {
      Re1 p = points_.chip<2>(it).chip<1>(is);
      if ((corners && B0((p.abs() <= thresh).all())()) || Norm(p / thresh) <= 1.f) {
        dsPoints.chip<2>(it).chip<1>(is) = p;
        for (int ii = 0; ii < 3; ii++) {
          p(ii) /= ratios(ii);
        }
        if (fullResTraces < 1 || it < fullResTraces) { // Ignore lo-res traces for this calculation
          minSamp = std::min(minSamp, is);
          maxSamp = std::max(maxSamp, is);
        }
      } else {
        dsPoints.chip<2>(it).chip<1>(is).setConstant(std::numeric_limits<float>::quiet_NaN());
      }
    }
  }
  Index const dsSamples = maxSamp + 1 - minSamp;
  Log::Print("Retaining samples {}-{}", minSamp, maxSamp);
  if (minSamp > maxSamp) { Log::Fail("No valid trajectory points remain after downsampling"); }
  dsPoints = Re3(dsPoints.slice(Sz3{0, minSamp, 0}, Sz3{3, dsSamples, nTraces()}));
  Log::Print("Downsampled trajectory dims {}", dsPoints.dimensions());
  return std::make_tuple(Trajectory(dsPoints, dsMatrix, dsVox), minSamp, dsSamples);
}

auto Trajectory::downsample(Cx5 const           &ks,
                            Eigen::Array3f const tgt,
                            Index const          fullResTraces,
                            bool const           shrink,
                            bool const           corners) const -> std::tuple<Trajectory, Cx5>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(tgt, fullResTraces, shrink, corners);
  Cx5 dsKs = ks.slice(Sz5{0, minSamp, 0, 0, 0}, Sz5{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3), ks.dimension(4)});
  return std::make_tuple(dsTraj, dsKs);
}

auto Trajectory::downsample(Cx4 const           &ks,
                            Eigen::Array3f const tgt,
                            Index const          fullResTraces,
                            bool const           shrink,
                            bool const           corners) const -> std::tuple<Trajectory, Cx4>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(tgt, fullResTraces, shrink, corners);
  Cx4 dsKs = ks.slice(Sz4{0, minSamp, 0, 0}, Sz4{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3)});
  return std::make_tuple(dsTraj, dsKs);
}

} // namespace rl
