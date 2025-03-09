#include "trajectory.hpp"

#include "log.hpp"
#include "tensors.hpp"

#include <cfenv>

namespace rl {

namespace {
constexpr Eigen::IndexPairList<Eigen::type2indexpair<0, 0>> matMul;
constexpr Eigen::IndexPairList<Eigen::type2indexpair<1, 0>> matMulT;
} // namespace

/* Temp Hack because .maximum() may be buggy on NEON */
template <int ND> auto GuessMatrix(Re3 const &points) -> Sz<ND>
{
  if (points.dimension(0) != ND) { throw Log::Failure("Traj", "Incorrect number of co-ordinates for GuessMatrix"); }
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

template <int ND> TrajectoryN<ND>::TrajectoryN(HD5::Reader &file, Array const voxel_size, SzN const matrix_size)
{
  points_ = file.readTensor<Re3>(HD5::Keys::Trajectory);
  if (std::all_of(matrix_size.cbegin(), matrix_size.cend(), [](Index ii) { return ii > 0; })) {
    matrix_ = matrix_size;
  } else if (file.exists(HD5::Keys::Trajectory, "matrix")) {
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
  if (nD != ND) { throw Log::Failure("Traj", "Points have {} co-ordinates, expected {}", nD, ND); }

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
    Log::Warn("Traj", "Discarded {} points ({:.2f}%) outside matrix", discarded, percent);
  }
  Log::Print("Traj", "{}D Samples {} Traces {} Matrix {} FOV {::.2f}", ND, nSamples(), nTraces(), matrix_, FOV());
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
  if (dims[1] != nSamples()) {
    throw Log::Failure("Traj", "Number of samples in data {} does not match trajectory {}", dims[1], nSamples());
  }
  if (dims[2] != nTraces()) {
    throw Log::Failure("Traj", "Number of traces in data {} does not match trajectory {}", dims[2], nTraces());
  }
}

template <int ND> auto TrajectoryN<ND>::compatible(TrajectoryN const &other) const -> bool
{
  if ((other.matrix() == matrix()) && (other.voxelSize() == voxelSize()).all()) {
    return true;
  } else {
    return false;
  }
}

template <int ND> auto TrajectoryN<ND>::matrix() const -> SzN { return matrix_; }

template <int ND> auto TrajectoryN<ND>::matrixForFOV(Array const fov) const -> SzN
{
  SzN matrix;
  for (Index ii = 0; ii < ND; ii++) {
    matrix[ii] = std::max(matrix_[ii], 2 * (Index)(fov[ii] / voxel_size_[ii] / 2.f));
  }
  Log::Print("Traj", "Requested FOV {::.1f}, matrix {}", fov.transpose(), matrix);
  return matrix;
}

template <int ND> auto TrajectoryN<ND>::matrixForFOV(Array const fov, Index const nB, Index const nT) const -> Sz<ND + 2>
{
  auto const m = this->matrixForFOV(fov);
  return AddBack(m, nB, nT);
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

template <int ND> void TrajectoryN<ND>::shiftInFOV(Eigen::Vector3f const shift, Cx5 &data) const
{
  Re1 delta(ND);
  for (Index ii = 0; ii < ND; ii++) {
    delta[ii] = shift[ii] / (voxel_size_[ii] * matrix_[ii]);
  }

  Log::Print("Traj", "Shifting FOV by {} {} {}", delta[0], delta[1], delta[2]);

  auto const shape = data.dimensions();

  // Check for NaNs (trajectory points that should be ignored) and zero the corresponding data points. Otherwise they become
  // NaN, and cause problems in iterative recons
  data.device(Threads::TensorDevice()) =
    data * points_.sum(Sz1{0})
             .isfinite()
             .reshape(Sz5{1, shape[1], shape[2], 1, 1})
             .broadcast(Sz5{shape[0], 1, 1, 1, shape[4]})
             .select((points_.contract(delta, matMul).template cast<Cx>() * Cx(0.f, 2.f * M_PI))
                       .exp()
                       .reshape(Sz5{1, shape[1], shape[2], 1, 1})
                       .broadcast(Sz5{shape[0], 1, 1, shape[3], shape[4]}),
                     data.constant(0.f));
}

template <int ND>
void TrajectoryN<ND>::shiftInFOV(Eigen::Vector3f const shift, Index const tst, Index const tsz, Cx5 &data) const
{
  Re1 delta(ND);
  for (Index ii = 0; ii < ND; ii++) {
    delta[ii] = shift[ii] / (voxel_size_[ii] * matrix_[ii]);
  }

  Log::Print("Traj", "Shifting traces {}-{} by {} {} {}", tst, tst + tsz - 1, delta[0], delta[1], delta[2]);

  auto const shape = data.dimensions();

  // Check for NaNs (trajectory points that should be ignored) and zero the corresponding data points. Otherwise they become
  // NaN, and cause problems in iterative recons
  Sz5 const dst{0, 0, tst, 0, 0};
  Sz5 const dsz = AddBack(FirstN<2>(data.dimensions()), tsz, data.dimension(3), data.dimension(4));
  auto      d = data.slice(dst, dsz);
  auto      p = points_.slice(Sz3{0, 0, tst}, Sz3{ND, shape[1], tsz});
  d.device(Threads::TensorDevice()) = d * p.sum(Sz1{0})
                                            .isfinite()
                                            .reshape(Sz4{1, shape[1], tsz, 1})
                                            .broadcast(Sz4{shape[0], 1, 1, 1})
                                            .select((p.contract(delta, matMul).template cast<Cx>() * Cx(0.f, 2.f * M_PI))
                                                      .exp()
                                                      .reshape(Sz4{1, shape[1], tsz, 1})
                                                      .broadcast(Sz4{shape[0], 1, 1, shape[3]}),
                                                    d.constant(0.f));
}

template <int ND> void TrajectoryN<ND>::moveInFOV(Eigen::Matrix<float, ND, ND> const R, Eigen::Vector3f const shift, Cx5 &data)
{
  Re2CMap const Rt(R.data(), Sz2{ND, ND});
  points_.device(Threads::TensorDevice()) = Re3(Rt.contract(points_, matMul));
  shiftInFOV(shift, data);
}

template <int ND>
void TrajectoryN<ND>::moveInFOV(
  Eigen::Matrix<float, ND, ND> const R, Eigen::Vector3f const s, Index const tst, Index const tsz, Cx5 &data)
{
  Re2CMap const Rt(R.data(), Sz2{ND, ND});
  auto          p = points_.slice(Sz3{0, 0, tst}, Sz3{ND, nSamples(), tsz});
  Log::Print("Traj", "Rotating traces {}-{} by\n{}", tst, tst + tsz - 1, fmt::streamed(Rt));
  p.device(Threads::TensorDevice()) = Re3(Rt.contract(p, matMul));
  shiftInFOV(s, tst, tsz, data);
}

template <int ND> auto TrajectoryN<ND>::points() const -> Re3 const & { return points_; }

template <int ND> auto TrajectoryN<ND>::point(int16_t const read, int32_t const spoke) const -> Eigen::Vector<float, ND>
{
  Re1 const                p = points_.template chip<2>(spoke).template chip<1>(read);
  Eigen::Vector<float, ND> pv;
  for (Index ii = 0; ii < ND; ii++) {
    pv[ii] = p(ii);
  }
  return pv;
}

auto FindFirstValidSample(Re3 const &traj)
{
  for (Index is = 0; is < traj.dimension(1); is++) {
    bool cont = false;
    for (Index it = 0; it < traj.dimension(2); it++) {
      for (Index id = 0; id < traj.dimension(0); id++) {
        if (traj(id, is, it) != traj(id, is, it)) {
          // Found an invalid trajectory point
          cont = true;
        }
      }
    }
    if (!cont) { return is; }
  }
  throw Log::Failure("traj", "No valid samples found");
}

auto FindLastValidSample(Re3 const &traj)
{
  for (Index is = traj.dimension(1) - 1; is >= 0; is--) {
    bool cont = false;
    for (Index it = 0; it < traj.dimension(2); it++) {
      for (Index id = 0; id < traj.dimension(0); id++) {
        if (traj(id, is, it) != traj(id, is, it)) {
          // Found an invalid trajectory point
          cont = true;
        }
      }
    }
    if (!cont) { return is; }
  }
  throw Log::Failure("traj", "No valid samples found");
}

template <int ND>
auto TrajectoryN<ND>::downsample(Array const tgtSize, bool const trim, bool const shrink, bool const corners) const
  -> std::tuple<TrajectoryN, Index, Index>
{
  Array ratios = voxel_size_ / tgtSize;
  if ((ratios > 1.f).any()) {
    throw Log::Failure("Traj", "Requested voxel-size {} is smaller than current {}", tgtSize, voxel_size_);
  }
  auto dsVox = voxel_size_;
  auto dsMatrix = matrix_;
  Re1  thresh(3);
  for (Index ii = 0; ii < ND; ii++) {
    if (shrink) {
      // Account for rounding
      dsMatrix[ii] = matrix_[ii] * ratios[ii];
      float const scale = static_cast<float>(matrix_[ii]) / dsMatrix[ii];
      ratios(ii) = 1.f / scale;
      dsVox[ii] = voxel_size_[ii] * scale;
    }
    thresh(ii) = matrix_[ii] * ratios(ii) / 2.f;
  }
  Log::Print("Traj", "Downsample {}->{} mm, matrix {}, ratios {}", voxel_size_, tgtSize, dsMatrix,
             fmt::streamed(ratios.transpose()));

  Re3 dsPoints(points_.dimensions());
  for (Index it = 0; it < nTraces(); it++) {
    for (Index is = 0; is < nSamples(); is++) {
      Re1 p = points_.template chip<2>(it).template chip<1>(is);
      if ((corners && B0((p.abs() <= thresh).all())()) || Norm<false>(p / thresh) <= 1.f) {
        dsPoints.chip<2>(it).chip<1>(is) = p;
        for (int ii = 0; ii < 3; ii++) {
          p(ii) /= ratios(ii);
        }
      } else {
        dsPoints.chip<2>(it).chip<1>(is).setConstant(std::numeric_limits<float>::quiet_NaN());
      }
    }
  }

  Index minSamp = 0;
  Index dsSamples = dsPoints.dimension(1);
  if (trim) {
    minSamp = FindFirstValidSample(dsPoints);
    Index const maxSamp = FindLastValidSample(dsPoints);
    dsSamples = maxSamp + 1 - minSamp;
    Log::Print("Traj", "Retaining samples {}-{}", minSamp, maxSamp);
    if (minSamp > maxSamp) { throw Log::Failure("Traj", "No valid trajectory points remain after downsampling"); }
    dsPoints = Re3(dsPoints.slice(Sz3{0, minSamp, 0}, Sz3{3, dsSamples, nTraces()}));
  }
  Log::Print("Traj", "Downsampled trajectory dims {}", dsPoints.dimensions());
  return std::make_tuple(TrajectoryN(dsPoints, dsMatrix, dsVox), minSamp, dsSamples);
}

template <int ND>
auto TrajectoryN<ND>::downsample(Cx5 const &ks, Array const tgt, bool const trim, bool const shrink, bool const corners) const
  -> std::tuple<TrajectoryN, Cx5>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(tgt, trim, shrink, corners);
  Cx5 dsKs = ks.slice(Sz5{0, minSamp, 0, 0, 0}, Sz5{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3), ks.dimension(4)});
  return std::make_tuple(dsTraj, dsKs);
}

template <int ND>
auto TrajectoryN<ND>::downsample(Cx4 const &ks, Array const tgt, bool const trim, bool const shrink, bool const corners) const
  -> std::tuple<TrajectoryN, Cx4>
{
  auto const [dsTraj, minSamp, nSamp] = downsample(tgt, trim, shrink, corners);
  Cx4 dsKs = ks.slice(Sz4{0, minSamp, 0, 0}, Sz4{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3)});
  return std::make_tuple(dsTraj, dsKs);
}

template <int ND>
auto TrajectoryN<ND>::downsample(Cx4 const &ks, Sz3 const tgtMat, bool const trim, bool const shrink, bool const corners) const
  -> std::tuple<TrajectoryN, Cx4>
{
  Array tgt;
  for (Index ii = 0; ii < ND; ii++) {
    tgt[ii] = (voxel_size_[ii] * matrix_[ii]) / tgtMat[ii];
  }
  auto const [dsTraj, minSamp, nSamp] = downsample(tgt, trim, shrink, corners);
  Cx4 dsKs = ks.slice(Sz4{0, minSamp, 0, 0}, Sz4{ks.dimension(0), nSamp, ks.dimension(2), ks.dimension(3)});
  return std::make_tuple(dsTraj, dsKs);
}

template <int ND> inline auto Sz2Array(Sz<ND> const &sz) -> Eigen::Array<float, ND, 1>
{
  Eigen::Array<float, ND, 1> a;
  for (Index ii = 0; ii < ND; ii++) {
    a[ii] = sz[ii];
  }
  return a;
}

template <int ND>
inline auto SubgridIndex(Eigen::Array<Index, ND, 1> const &sg, Eigen::Array<Index, ND, 1> const &ngrids) -> Index
{
  Index ind = 0;
  Index stride = 1;
  for (Index ii = 0; ii < ND; ii++) {
    ind += stride * sg[ii];
    stride *= ngrids[ii];
  }
  return ind;
}

template <int ND>
auto TrajectoryN<ND>::toCoordLists(Sz<ND> const &oshape, Index const kW, Index const sgSz, bool const conj) const
  -> std::vector<CoordList>
{
  std::fesetround(FE_TONEAREST);
  std::vector<Coord> coords;

  using Arrayf = Coord::template Array<float>;
  using Arrayi = Coord::template Array<Index>;

  Arrayf const mat = Sz2Array(this->matrix());
  Arrayf const omat = Sz2Array(oshape);
  Arrayf const osamp = omat / mat;
  Log::Print("Traj", "Nominal matrix {} grid matrix {} over-sampling {::.2f}", mat.transpose(), omat.transpose(),
             osamp.transpose());

  Arrayf const k0 = omat / 2;
  Index        valid = 0;
  Index        invalids = 0;
  Arrayi const nSubgrids = (omat / sgSz).ceil().template cast<Index>();
  Index const  nTotal = nSubgrids.prod();

  std::vector<CoordList> subs(nTotal);
  for (int32_t it = 0; it < this->nTraces(); it++) {
    for (int16_t is = 0; is < this->nSamples(); is++) {
      Arrayf const p = this->point(is, it) * (conj ? -1.f : 1.f);
      if ((p != p).any()) {
        invalids++;
        continue;
      }
      Arrayf const k = p * osamp + k0;
      Arrayf const ki = k.unaryExpr([](float const &e) { return std::nearbyint(e); });
      if ((ki < 0.f).any() || (ki >= omat).any()) {
        invalids++;
        continue;
      }
      Arrayf const ko = k - ki;
      Arrayi const ksub = (ki / sgSz).floor().template cast<Index>();
      Arrayi const kint = ki.template cast<Index>() - (ksub * sgSz) + (kW / 2);
      Index const  sgind = SubgridIndex(ksub, nSubgrids);
      subs[sgind].corner = ksub.template cast<int16_t>();
      subs[sgind].coords.push_back(Coord{.cart = kint.template cast<int16_t>(), .sample = is, .trace = it, .offset = ko});
      valid++;
    }
  }
  Log::Print("Traj", "Ignored {} invalid trajectory points, {} remaing", invalids, valid);
  auto const eraseCount = std::erase_if(subs, [](auto const &s) { return s.coords.empty(); });
  Log::Debug("Traj", "Removed {} empty subgrids, {} remaining", eraseCount, subs.size());
  Log::Debug("Traj", "Sorting subgrids");
  std::sort(subs.begin(), subs.end(), [](CoordList const &a, CoordList const &b) { return a.coords.size() > b.coords.size(); });
  Log::Debug("Traj", "Sorting coords");
  for (auto &s : subs) {
    std::sort(s.coords.begin(), s.coords.end(), [](Coord const &a, Coord const &b) {
      // Compare on ijk location
      for (size_t di = 0; di < ND; di++) {
        size_t id = ND - 1 - di;
        if (a.cart[id] < b.cart[id]) {
          return true;
        } else if (b.cart[id] < a.cart[id]) {
          return false;
        }
      }
      return false;
    });
  }

  return subs;
}

template struct TrajectoryN<1>;
template struct TrajectoryN<2>;
template struct TrajectoryN<3>;

} // namespace rl
