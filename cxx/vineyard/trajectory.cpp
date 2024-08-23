#include "trajectory.hpp"

#include "log.hpp"
#include "tensors.hpp"

namespace rl {

/* Temp Hack because .maximum() may be buggy on NEON */
template <int ND> auto GuessMatrix(Re3 const &points) -> Sz<ND>
{
  if (points.dimension(0) != ND) { Log::Fail("Incorrect number of co-ordinates for GuessMatrix"); }
  Re1 max(ND);
  max.setZero();
  for (Index ii = 0; ii < points.dimension(1); ii++) {
    for (Index ij = 0; ij < points.dimension(2); ij++) {
      for (Index ic = 0; ic < ND; ic++) {
        auto const a = std::fabs(points(ic, ii, ij));
        if (a > max(ic)) { max(ic) = a; }
      }
    }
  }
  Sz<ND> mat;
  for (Index ii = 0; ii < ND; ii++) {
    mat[ii] = std::max((Index)(std::ceil(max(ii)) * 2), 1L);
  }
  return mat;
}

template <int ND>
TrajectoryN<ND>::TrajectoryN(Re3 const &points, Array const voxel_size)
  : points_{points}
  , matrix_{GuessMatrix<ND>(points_)}
  , voxel_size_{voxel_size}
{
  init();
}

template <int ND>
TrajectoryN<ND>::TrajectoryN(Re3 const &points, SzN const matrix, Array const voxel_size)
  : points_{points}
  , matrix_{matrix}
  , voxel_size_{voxel_size}
{
  init();
}

template <int ND> TrajectoryN<ND>::TrajectoryN(HD5::Reader &file, Array const voxel_size)
{
  points_ = file.readTensor<Re3>(HD5::Keys::Trajectory);
  if (file.exists(HD5::Keys::Trajectory, "matrix")) {
    matrix_ = file.readAttributeSz<ND>(HD5::Keys::Trajectory, "matrix");
  } else {
    matrix_ = GuessMatrix<ND>(points_);
  }
  voxel_size_ = voxel_size;
  init();
}

template <int ND> void TrajectoryN<ND>::init()
{
  Index const nD = points_.dimension(0);
  if (nD != ND) { Log::Fail("Trajectory points have {} co-ordinates, expected {}", nD, ND); }

  Index discarded = 0;
  Re1   mat(nD);
  for (Index ii = 0; ii < ND; ii++) {
    mat(ii) = matrix_[ii] / 2.f;
  }
  for (Index is = 0; is < points_.dimension(2); is++) {
    for (Index ir = 0; ir < points_.dimension(1); ir++) {
      if (B0((points_.template chip<2>(is).template chip<1>(ir).abs() > mat).any())()) {
        points_.template chip<2>(is).template chip<1>(ir).setConstant(std::numeric_limits<float>::quiet_NaN());
        discarded++;
      }
    }
  }
  if (discarded > 0) {
    Index const total = points_.dimension(1) * points_.dimension(2);
    float const percent = (100.f * discarded) / total;
    Log::Warn("Discarded {} trajectory points ({:.2f}%) outside matrix", discarded, percent);
  }
  Log::Print("{}D Trajectory Samples {} Traces {} Matrix {} FOV {}", ND, nSamples(), nTraces(), matrix_, FOV());
}

template <int ND> void TrajectoryN<ND>::write(HD5::Writer &file) const
{
  file.writeTensor(HD5::Keys::Trajectory, points_.dimensions(), points_.data(), HD5::Dims::Trajectory);
  file.writeAttribute(HD5::Keys::Trajectory, "matrix", matrix_);
}

template <int ND> auto TrajectoryN<ND>::nSamples() const -> Index { return points_.dimension(1); }

template <int ND> auto TrajectoryN<ND>::nTraces() const -> Index { return points_.dimension(2); }

template <int ND> void TrajectoryN<ND>::checkDims(SzN const dims) const
{
  if (dims[1] != nSamples()) { Log::Fail("Number of samples in data {} does not match trajectory {}", dims[1], nSamples()); }
  if (dims[2] != nTraces()) { Log::Fail("Number of traces in data {} does not match trajectory {}", dims[2], nTraces()); }
}

template <int ND> auto TrajectoryN<ND>::compatible(TrajectoryN const &other) const -> bool
{
  if ((other.matrix() == matrix()) && (other.voxelSize() == voxelSize()).all()) {
    return true;
  } else {
    return false;
  }
}

template <int ND> auto TrajectoryN<ND>::matrix(float const os) const -> SzN
{
  if (os == 1.f) {
    return matrix_;
  } else {
    SzN om;
    for (Index ii = 0; ii < ND; ii++) {
      om[ii] = os * matrix_[ii];
    }
    return om;
  }
}

template <int ND> auto TrajectoryN<ND>::matrixForFOV(Array const fov, float const os) const -> SzN
{
  SzN matrix;
  for (Index ii = 0; ii < ND; ii++) {
    matrix[ii] = os * std::max(matrix_[ii], 2 * (Index)(fov[ii] / voxel_size_[ii] / 2.f));
  }
  Log::Print("Trajectory FOV {} matrix {}. Requested FOV {} oversampling {} matrix {}", FOV().transpose(), matrix_,
             fov.transpose(), os, matrix);
  return matrix;
}

template <int ND>
auto TrajectoryN<ND>::matrixForFOV(Array const fov, Index const nB, Index const nT, float const os) const -> Sz<ND + 2>
{
  Sz<ND + 2> matrix;
  for (Index ii = 0; ii < ND; ii++) {
    matrix[ii + 1] = os * std::max(matrix_[ii], 2 * (Index)(fov[ii] / voxel_size_[ii] / 2.f));
  }
  matrix[0] = nB;
  matrix[ND + 1] = nT;
  Log::Print("Trajectory FOV {} matrix {}. Requested FOV {} oversampling {} matrix {}", FOV().transpose(), matrix_,
             fov.transpose(), os, matrix);
  return matrix;
}

template <int ND> auto TrajectoryN<ND>::voxelSize() const -> Array { return voxel_size_; }

template <int ND> auto TrajectoryN<ND>::FOV() const -> Array
{
  Array fov;
  for (Index ii = 0; ii < ND; ii++) {
    fov[ii] = matrix_[ii] * voxel_size_[ii];
  }
  return fov;
}

template <int ND> void TrajectoryN<ND>::shiftFOV(Eigen::Vector3f const shift, Cx5 &data)
{
  Re1 delta(ND);
  for (Index ii = 0; ii < ND; ii++) {
    delta[ii] = shift[ii] / (voxel_size_[ii] * matrix_[ii]);
  }

  Eigen::IndexPairList<Eigen::type2indexpair<0, 0>> zero2zero;
  Log::Print("Shifting FOV by {} {} {}", delta[0], delta[1], delta[2]);

  auto const shape = data.dimensions();

  // Check for NaNs (trajectory points that should be ignored) and zero the corresponding data points. Otherwise they become
  // NaN, and cause problems in iterative recons
  data.device(Threads::GlobalDevice()) =
    data * points_.sum(Sz1{0})
             .isfinite()
             .reshape(Sz5{1, shape[1], shape[2], 1, 1})
             .broadcast(Sz5{shape[0], 1, 1, 1, shape[4]})
             .select((points_.contract(delta, zero2zero).template cast<Cx>() * Cx(0.f, 2.f * M_PI))
                       .exp()
                       .reshape(Sz5{1, shape[1], shape[2], 1, 1})
                       .broadcast(Sz5{shape[0], 1, 1, shape[3], shape[4]}),
                     data.constant(0.f));
}

template <int ND> Re3 const &TrajectoryN<ND>::points() const { return points_; }

template <int ND> Re1 TrajectoryN<ND>::point(int16_t const read, int32_t const spoke) const
{
  Re1 const p = points_.template chip<2>(spoke).template chip<1>(read);
  return p;
}

template <int ND>
auto TrajectoryN<ND>::downsample(Array const tgtSize, Index const fullResTraces, bool const shrink, bool const corners) const
  -> std::tuple<TrajectoryN, Index, Index>
{
  Array ratios = voxel_size_ / tgtSize;
  if ((ratios > 1.f).any()) {
    Log::Fail("Downsample voxel-size {} is larger than current voxel-size {}", tgtSize, voxel_size_);
  }
  auto dsVox = voxel_size_;
  auto dsMatrix = matrix_;
  Re1  thresh(3);
  for (Index ii = 0; ii < ND; ii++) {
    if (shrink) {
      // Account for rounding
      dsMatrix[ii] = matrix_[ii] * ratios[ii];
      float const scale = static_cast<float>(matrix_[0]) / dsMatrix[0];
      ratios(ii) = 1.f / scale;
      dsVox[ii] = voxel_size_[ii] * scale;
    }
    thresh(ii) = matrix_[ii] * ratios(ii) / 2.f;
  }
  Log::Print("Downsample {}->{} mm, matrix {}, ratios {}", voxel_size_, tgtSize, dsMatrix, fmt::streamed(ratios.transpose()));

  Index minSamp = nSamples(), maxSamp = 0;
  Re3   dsPoints(points_.dimensions());
  for (Index it = 0; it < nTraces(); it++) {
    for (Index is = 0; is < nSamples(); is++) {
      Re1 p = points_.template chip<2>(it).template chip<1>(is);
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
  return std::make_tuple(TrajectoryN(dsPoints, dsMatrix, dsVox), minSamp, dsSamples);
}

template <int ND>
auto TrajectoryN<ND>::downsample(
  Cx5 const &ks, Array const tgt, Index const fullResTraces, bool const shrink, bool const corners) const
  -> std::tuple<TrajectoryN, Cx5>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(tgt, fullResTraces, shrink, corners);
  Cx5 dsKs = ks.slice(Sz5{0, minSamp, 0, 0, 0}, Sz5{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3), ks.dimension(4)});
  return std::make_tuple(dsTraj, dsKs);
}

template <int ND>
auto TrajectoryN<ND>::downsample(
  Cx4 const &ks, Array const tgt, Index const fullResTraces, bool const shrink, bool const corners) const
  -> std::tuple<TrajectoryN, Cx4>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(tgt, fullResTraces, shrink, corners);
  Cx4 dsKs = ks.slice(Sz4{0, minSamp, 0, 0}, Sz4{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3)});
  return std::make_tuple(dsTraj, dsKs);
}

template <int ND>
auto TrajectoryN<ND>::downsample(
  Cx4 const &ks, Sz3 const tgtMat, Index const fullResTraces, bool const shrink, bool const corners) const
  -> std::tuple<TrajectoryN, Cx4>
{
  Array tgt;
  for (Index ii = 0; ii < ND; ii++) {
    tgt[ii] = (voxel_size_[ii] * matrix_[ii]) / tgtMat[ii];
  }
  auto const [dsTraj, minSamp, nSamp] = downsample(tgt, fullResTraces, shrink, corners);
  Cx4 dsKs = ks.slice(Sz4{0, minSamp, 0, 0}, Sz4{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3)});
  return std::make_tuple(dsTraj, dsKs);
}

template struct TrajectoryN<1>;
template struct TrajectoryN<2>;
template struct TrajectoryN<3>;

} // namespace rl
