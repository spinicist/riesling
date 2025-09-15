#include "trajectory.hpp"

#include "log/log.hpp"
#include "tensors.hpp"

#include <cfenv>
#include <flux.hpp>

namespace rl {

namespace {
constexpr Eigen::IndexPairList<Eigen::type2indexpair<0, 0>> matMul;
} // namespace

/* Temp Hack because .maximum() may be buggy on NEON */
template <int ND> auto GuessMatrix(Re3 const &points) -> Sz<ND>
{
  if (points.dimension(0) != ND) { throw Log::Failure("Traj", "Incorrect number of co-ordinates for GuessMatrix"); }
  std::array<float, ND> max;
  std::fill_n(max.begin(), ND, 0.f);
  for (Index ii = 0; ii < points.dimension(1); ii++) {
    for (Index ij = 0; ij < points.dimension(2); ij++) {
      for (Index ic = 0; ic < ND; ic++) {
        auto const a = std::fabs(points(ic, ii, ij));
        if (a > max[ic]) { max[ic] = a; }
      }
    }
  }
  Sz<ND> mat;
  for (Index ii = 0; ii < ND; ii++) {
    mat[ii] = std::max((Index)(std::ceil(max[ii]) * 2), 1L);
  }
  return mat;
}

template <int ND> TrajectoryN<ND>::TrajectoryN(Re3 const &points, Array const voxel_size)
  : points_{points}
  , matrix_{GuessMatrix<ND>(points_)}
  , voxel_size_{voxel_size}
{
  init();
}

template <int ND> TrajectoryN<ND>::TrajectoryN(Re3 const &points, SzN const matrix, Array const voxel_size)
  : points_{points}
  , matrix_{matrix}
  , voxel_size_{voxel_size}
{
  init();
}

template <int ND> TrajectoryN<ND>::TrajectoryN(HD5::Reader &file, Array const voxel_size, SzN const mat)
{
  points_ = file.readTensor<Re3>(HD5::Keys::Trajectory);
  voxel_size_ = voxel_size;
  if (points_.dimension(0) != ND) {
    throw(Log::Failure("Traj", "Trajectory on disk was {}D, expected {}D", points_.dimension(0), ND));
  }
  if (file.exists(HD5::Keys::Trajectory, "matrix")) {
    matrix_ = file.readAttributeShape<ND>(HD5::Keys::Trajectory, "matrix");
  } else {
    matrix_ = GuessMatrix<ND>(points_);
  }
  /* If the matrix size is overridden, adjust the voxel size as well*/
  if (std::all_of(mat.cbegin(), mat.cend(), [](Index ii) { return ii > 0; })) {
    Array matO, matN;
    std::copy_n(matrix_.begin(), ND, matO.begin());
    std::copy_n(mat.begin(), ND, matN.begin());
    Array ratio = matO / matN;
    voxel_size_ = voxel_size * ratio;
    matrix_ = mat;
  } 
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
  file.writeAttribute(HD5::Keys::Trajectory, "matrix", ToArray(matrix_));
}

template <int ND> auto TrajectoryN<ND>::nSamples() const -> Index { return points_.dimension(1); }

template <int ND> auto TrajectoryN<ND>::nTraces() const -> Index { return points_.dimension(2); }

template <int ND> void TrajectoryN<ND>::checkDims(Sz3 const dims) const
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
  if (flux::all(fov, [](float f) { return f == 0.f; })) {
    Log::Print("Traj", "Nominal FOV, matrix {}", matrix_);
    return matrix_;
  } else {
    SzN matrix;
    for (Index ii = 0; ii < ND; ii++) {
      matrix[ii] = 2 * (Index)(fov[ii] / voxel_size_[ii] / 2.f);
    }
    Log::Print("Traj", "FOV {::.1f} => matrix {}", fov.transpose(), matrix);
    return matrix;
  }
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
  shiftInFOV(shift, 0, data.dimension(2), data);
}

template <int ND>
void TrajectoryN<ND>::shiftInFOV(Eigen::Vector3f const shift, Index const tst, Index const tsz, Cx5 &data) const
{
  Re1 delta(ND);
  for (Index ii = 0; ii < ND; ii++) {
    delta[ii] = shift[ii] / (voxel_size_[ii] * matrix_[ii]);
  }

  Log::Print("Traj", "Traces {}-{} FOV shift {} mm, fraction {}", tst, tst + tsz - 1, fmt::streamed(shift.transpose()),
             fmt::streamed(delta));
  Sz5 const dshape = data.dimensions();
  Sz5 const rshape = Sz5{1, dshape[1], tsz, 1, 1};
  Sz5 const bshape = Sz5{dshape[0], 1, 1, dshape[3], dshape[4]};
  // Check for NaNs (trajectory points that should be ignored) and zero the corresponding data points. Otherwise they become
  // NaN, and cause problems in iterative recons
  auto d = data.slice(Sz5{0, 0, tst, 0, 0}, AddBack(FirstN<2>(data.dimensions()), tsz, dshape[3], dshape[4]));
  auto p = points_.slice(Sz3{0, 0, tst}, Sz3{ND, dshape[1], tsz});
  auto mask = p.sum(Sz1{0}).isfinite().reshape(rshape).broadcast(bshape);
  d.device(Threads::TensorDevice()) = mask.select(
    d * (p.contract(delta, matMul).template cast<Cx>() * Cx(0.f, 2.f * M_PI)).exp().reshape(rshape).broadcast(bshape),
    d.constant(0.f));
}

template <int ND> void TrajectoryN<ND>::moveInFOV(Eigen::Matrix<float, ND, ND> const R, Eigen::Vector3f const shift, Cx5 &data)
{
  moveInFOV(R, shift, 0, data.dimension(2), data);
}

template <int ND> void TrajectoryN<ND>::moveInFOV(
  Eigen::Matrix<float, ND, ND> const R, Eigen::Vector3f const s, Index const tst, Index const tsz, Cx5 &data)
{
  if (tst + tsz > nTraces()) { throw Log::Failure("Traj", "Max trace {} exceeded number of traces {}", tst + tsz, nTraces()); }
  Re2CMap Rt(R.data(), Sz2{ND, ND});
  auto    p = points_.slice(Sz3{0, 0, tst}, Sz3{ND, nSamples(), tsz});
  Log::Debug("Traj", "Rotating traces {}-{}", tst, tst + tsz - 1);
  p.device(Threads::TensorDevice()) = Re3(Rt.contract(p, matMul));
  shiftInFOV(s, tst, tsz, data);
}

template <int ND> auto TrajectoryN<ND>::points() const -> Re3 const & { return points_; }

template <int ND> auto TrajectoryN<ND>::point(int32_t const read, int32_t const spoke) const -> Eigen::Vector<float, ND>
{
  Re1 const                p = points_.template chip<2>(spoke).template chip<1>(read);
  Eigen::Vector<float, ND> pv;
  for (Index ii = 0; ii < ND; ii++) {
    pv[ii] = p(ii);
  }
  return pv;
}

template <int ND> void TrajectoryN<ND>::downsample(Array const tgtSize, bool const shrinkMatrix, bool const keepCorners)
{
  Re1 thresh(ND);
  for (Index ii = 0; ii < ND; ii++) {
    float ratio = voxel_size_[ii] / tgtSize[ii];
    if ((ratio > 1.f)) {
      throw Log::Failure("Traj", "Requested voxel-size {} is smaller than current {}", tgtSize, voxel_size_);
    }
    if (shrinkMatrix) { // Account for rounding
      Index const m = matrix_[ii] * ratio;
      ratio = (1.f * m) / matrix_[ii];
      voxel_size_[ii] /= ratio;
      matrix_[ii] = m;
      thresh(ii) = matrix_[ii] / 2.f;
    } else {
      thresh(ii) = matrix_[ii] * ratio / 2.f;
    }
  }
  Log::Print("Traj", "Downsample {} voxel-size, {} matrix, k-space threshold {}", voxel_size_, matrix_, fmt::streamed(thresh));

  for (Index it = 0; it < nTraces(); it++) {
    for (Index is = 0; is < nSamples(); is++) {
      Re1 p = points_.template chip<2>(it).template chip<1>(is);
      if ((keepCorners && B0((p.abs() > thresh).all())()) || Norm<false>(p / thresh) > 1.f) {
        points_.template chip<2>(it).template chip<1>(is).setConstant(std::numeric_limits<float>::quiet_NaN());
      }
    }
  }
}

struct ST
{
  Index s;
  Index t;
};

auto FindFirstValid(Re3 const &traj) -> ST
{
  Index s = traj.dimension(1), t = traj.dimension(2);
  for (Index it = 0; it < traj.dimension(2); it++) {
    for (Index is = 0; is < traj.dimension(1); is++) {
      if (std::isfinite(Sum(traj.chip<2>(it).chip<1>(is)))) {
        s = std::min(s, is);
        t = std::min(t, it);
      }
    }
  }
  if (s == traj.dimension(1) || t == traj.dimension(2)) {
    throw Log::Failure("traj", "No valid trajectory points found");
  } else {
    return {s, t};
  }
}

auto FindFirstInvalidSample(Re3 const &traj) -> ST
{
  Index s = traj.dimension(1), t = traj.dimension(2);
  for (Index it = 0; it < traj.dimension(2); it++) {
    for (Index is = 0; is < traj.dimension(1); is++) {
      if (!std::isfinite(Sum(traj.chip<2>(it).chip<1>(is)))) {
        s = std::min(s, is);
      }
    }
  }
  if (t < 0) {
    throw Log::Failure("traj", "No valid trajectory points found");
  } else {
    return {s, t};
  }
}

auto FindLastValid(Re3 const &traj) -> ST
{
  Index s = -1, t = -1;
  for (Index it = traj.dimension(2) - 1; it >= 0; it--) {
    for (Index is = traj.dimension(1) - 1; is >= 0; is--) {
      if (std::isfinite(Sum(traj.chip<2>(it).chip<1>(is)))) {
        s = std::max(s, is);
        t = std::max(t, it);
      }
    }
  }
  if (s == -1 || t == -1) {
    throw Log::Failure("traj", "No valid trajectory points found");
  } else {
    return {s, t};
  }
}

template <int ND> template <int D> auto TrajectoryN<ND>::trim(CxNCMap<D> const ks, bool const aggressive) -> CxN<D>
{
  auto const min = FindFirstValid(points_);
  auto const max = aggressive ? FindFirstInvalidSample(points_) : FindLastValid(points_);
  Log::Print("Traj", "Retaining samples {}-{} traces {}-{}", min.s, max.s, min.t, max.t);
  Index const nS = max.s - min.s;
  Index const nT = max.t - min.t;
  points_ = Re3(points_.slice(Sz3{0, min.s, min.t}, Sz3{ND, nS, nT}));
  return ks.slice(AddFront(Sz<D - 3>{}, 0, min.s, min.t), AddFront(LastN<D - 3>(ks.dimensions()), ks.dimension(0), nS, nT));
}

template <int ND> template <int D> auto TrajectoryN<ND>::trim(CxN<D> const &ks, bool const aggressive) -> CxN<D>
{
  return trim(CxNCMap<D>(ks.data(), ks.dimensions()), aggressive);
}

template <int ND> inline auto Sz2Array(Sz<ND> const &sz) -> Eigen::Array<float, ND, 1>
{
  Eigen::Array<float, ND, 1> a;
  for (Index ii = 0; ii < ND; ii++) {
    a[ii] = sz[ii];
  }
  return a;
}

template <int ND> inline auto SubgridIndex(Eigen::Array<Index, ND, 1> const &sg, Eigen::Array<Index, ND, 1> const &ngrids)
  -> Index
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
    for (int32_t is = 0; is < this->nSamples(); is++) {
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
  Log::Print("Traj", "Ignored {} invalid trajectory points, {} remain", invalids, valid);
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

template auto TrajectoryN<2>::trim(Cx3CMap, bool const) -> Cx3;
template auto TrajectoryN<2>::trim(Cx4CMap, bool const) -> Cx4;
template auto TrajectoryN<2>::trim(Cx5CMap, bool const) -> Cx5;

template auto TrajectoryN<2>::trim(Cx3 const &, bool const) -> Cx3;
template auto TrajectoryN<2>::trim(Cx4 const &, bool const) -> Cx4;
template auto TrajectoryN<2>::trim(Cx5 const &, bool const) -> Cx5;

template auto TrajectoryN<3>::trim(Cx3CMap, bool const) -> Cx3;
template auto TrajectoryN<3>::trim(Cx4CMap, bool const) -> Cx4;
template auto TrajectoryN<3>::trim(Cx5CMap, bool const) -> Cx5;

template auto TrajectoryN<3>::trim(Cx3 const &, bool const) -> Cx3;
template auto TrajectoryN<3>::trim(Cx4 const &, bool const) -> Cx4;
template auto TrajectoryN<3>::trim(Cx5 const &, bool const) -> Cx5;

} // namespace rl
