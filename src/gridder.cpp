#include "gridder.h"

#include "fft3.h"
#include "filter.h"
#include "tensorOps.h"
#include "threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

template <typename T>
inline decltype(auto) nearby(T &&x)
{
  return x.array().unaryExpr([](float const &e) { return std::nearbyint(e); });
}

// Helper function to get a "good" FFT size. Empirical rule of thumb - multiples of 8 work well
inline long fft_size(float const x)
{
  if (x > 8.f) {
    return (std::lrint(x) + 7L) & ~7L;
  } else {
    return (long)std::ceil(x);
  }
}

// Helper function to convert Tensor to Point
inline Point3 toCart(R1 const &p, float const xyScale, float const zScale)
{
  return Point3{p(0) * xyScale, p(1) * xyScale, p(2) * zScale};
}

Gridder::Gridder(
    Info const &info,
    R3 const &traj,
    float const os,
    Kernel *const kernel,
    bool const fastgrid,
    Log &log,
    float const res,
    bool const shrink)
    : info_{info}
    , oversample_{os}
    , DCexp_{1.f}
    , kernel_{kernel}
    , safe_{!fastgrid}
    , log_{log}
{
  assert(traj.dimension(0) == 3);
  assert(traj.dimension(1) == info_.read_points);
  assert(traj.dimension(2) == info_.spokes_total());

  // Work out grid dimensions for the given resolution and oversampling
  float const ratio = (res > 0.f) ? info_.voxel_size.minCoeff() / res : 1.f;
  long const nomSz = fft_size(oversample_ * info_.matrix.maxCoeff());
  long const nomRad = nomSz / 2 - 1;
  long const gridSz = shrink ? fft_size(nomSz * ratio) : nomSz;
  if (info.type == Info::Type::ThreeDStack) {
    dims_ = {gridSz, gridSz, info_.matrix[2]};
  } else {
    dims_ = {gridSz, gridSz, gridSz};
  }
  long const maxRad = (shrink ? (gridSz / 2 - 1) : std::lrint(nomRad * ratio)) - kernel_->radius();
  log_.info(
      FMT_STRING("Gridder: Dimensions {} Resolution {} Ratio {} maxRad {}"),
      dims_,
      res,
      ratio,
      maxRad);

  auto const hires = spokeInfo(traj.chip(info_.spokes_lo, 2), nomRad, maxRad, 1.f);
  log_.info(FMT_STRING("Hi-res spokes using points {}-{}"), hires.lo, hires.hi);
  coords_ = genCoords(traj, info_.spokes_lo, info_.spokes_hi, hires);
  if (info_.spokes_lo) {
    // Only grid lo-res to where hi-res begins (i.e. fill the dead-time gap)
    float const spokeOversamp = info_.read_points / (info_.matrix.maxCoeff() / 2);
    float const max_r = info_.read_gap * oversample_ / spokeOversamp;
    auto const lores = spokeInfo(traj.chip(0, 2), nomRad, max_r, info_.lo_scale);
    log_.info(FMT_STRING("Lo-res spokes using points {}-{}"), lores.lo, lores.hi);
    auto const temp = genCoords(traj, 0, info_.spokes_lo, lores);
    coords_.insert(coords_.end(), temp.begin(), temp.end());
  }
  sortCoords();
}

// Helper function to find the points along a spoke we will actually use
Gridder::SpokeInfo_t
Gridder::spokeInfo(R2 const &traj, long const nomRad, float const maxRad, float const scale)
{
  SpokeInfo_t s;
  s.xy = nomRad / scale;
  s.z = (info_.type == Info::Type::ThreeD) ? s.xy : 1.f;
  for (int16_t ir = info_.read_gap; ir < info_.read_points; ir++) {
    float const rad = toCart(traj.chip(ir, 1), s.xy, s.z).norm();
    if (rad <= maxRad) { // Discard points above the desired resolution
      s.lo = std::min(ir, s.lo);
      s.hi = std::max(ir, s.hi);
    }
  }
  return s;
}

std::vector<Gridder::Coords>
Gridder::genCoords(R3 const &traj, int32_t const spoke0, long const spokeSz, SpokeInfo_t const &s)
{
  long const readSz = s.hi - s.lo + 1;
  long const totalSz = readSz * spokeSz;
  std::vector<Coords> coords(totalSz);

  std::fesetround(FE_TONEAREST);
  Size3 const center(dims_[0] / 2, dims_[1] / 2, dims_[2] / 2);
  auto coordTask = [&](long const lo_spoke, long const hi_spoke) {
    long index = lo_spoke * readSz;
    for (int32_t is = lo_spoke; is < hi_spoke; is++) {
      for (int16_t ir = s.lo; ir <= s.hi; ir++) {
        NoncartesianIndex const nc{.spoke = is + spoke0, .read = ir};
        R1 const tp = traj.chip(nc.spoke, 2).chip(nc.read, 1);
        Point3 const xyz = toCart(tp, s.xy, s.z);
        Size3 const gp = nearby(xyz).cast<int16_t>();
        Point3 const offset = xyz - gp.cast<float>().matrix();
        Size3 const cart = gp + center;
        coords[index++] = Coords{.cart = CartesianIndex{cart(0), cart(1), cart(2)},
                                 .noncart = nc,
                                 .sdc = 1.f,
                                 .offset = offset};
      }
    }
  };
  auto start = log_.now();
  Threads::RangeFor(coordTask, spokeSz);
  log_.debug("Grid co-ord calcs: {}", log_.toNow(start));
  return coords;
}

void Gridder::sortCoords()
{
  auto const start = log_.now();
  sortedIndices_.resize(coords_.size());
  std::iota(sortedIndices_.begin(), sortedIndices_.end(), 0);
  std::sort(sortedIndices_.begin(), sortedIndices_.end(), [&](long const a, long const b) {
    auto const &ac = coords_[a].cart;
    auto const &bc = coords_[b].cart;
    return (ac.z < bc.z) ||
           ((ac.z == bc.z) && ((ac.y < bc.y) || ((ac.y == bc.y) && (ac.x < bc.x))));
  });
  log_.debug("Grid co-ord sorting: {}", log_.toNow(start));
}

Dims3 Gridder::gridDims() const
{
  return dims_;
}

void Gridder::setSDC(float const d)
{
  for (auto &c : coords_) {
    c.sdc = d;
  }
}

void Gridder ::setSDC(R2 const &sdc)
{
  for (auto &c : coords_) {
    if (c.noncart.read >= sdc.dimension(0) || c.noncart.spoke >= sdc.dimension(1)) {
      log_.fail("SDC dimensions {} do not match trajectory", fmt::join(sdc.dimensions(), ","));
    }
    c.sdc = sdc(c.noncart.read, c.noncart.spoke);
  }
}

void Gridder::setSDCExponent(float const dce)
{
  DCexp_ = dce;
}

void Gridder::setUnsafe()
{
  safe_ = true;
}

void Gridder::setSafe()
{
  safe_ = false;
}

Cx4 Gridder::newGrid() const
{
  Cx4 g(info_.channels, dims_[0], dims_[1], dims_[2]);
  g.setZero();
  return g;
}

Cx3 Gridder::newGrid1() const
{
  Cx3 g(dims_);
  g.setZero();
  return g;
}

void Gridder::toCartesian(Cx2 const &noncart, Cx3 &cart) const
{
  assert(noncart.dimension(0) == info_.read_points);
  assert(noncart.dimension(1) == info_.spokes_total());
  assert(cart.dimension(0) == dims_[0]);
  assert(cart.dimension(1) == dims_[1]);
  assert(cart.dimension(2) == dims_[2]);
  assert(sortedIndices_.size() == coords_.size());
  auto const st = kernel_->start();
  auto const sz = kernel_->size();

  auto dev = Threads::GlobalDevice();
  long const nThreads = dev.numThreads();
  std::vector<Cx3> workspace(nThreads);
  std::vector<long> minZ(nThreads, 0L), szZ(nThreads, 0L);
  auto grid_task = [&](long const lo, long const hi, long const ti) {
    // Allocate working space for this thread

    minZ[ti] = coords_[sortedIndices_[lo]].cart.z - kernel_->radius();
    if (safe_) {
      long const maxZ = coords_[sortedIndices_[hi - 1]].cart.z + kernel_->radius() + 1;
      szZ[ti] = maxZ - minZ[ti];
      workspace[ti].resize(cart.dimension(0), cart.dimension(1), szZ[ti]);
      workspace[ti].setZero();
    }

    for (auto ii = lo; ii < hi; ii++) {
      if (lo == 0) {
        log_.progress(ii, hi);
      }
      auto const &cp = coords_[sortedIndices_[ii]];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      auto const &dc = std::complex<float>(pow(cp.sdc, DCexp_), 0.f);
      Cx3 const vals = noncart(nc.read, nc.spoke) * kernel_->kspace(cp.offset).cast<Cx>() * dc;
      if (safe_) {
        workspace[ti].slice(Sz3{c.x + st[0], c.y + st[1], c.z + st[2] - minZ[ti]}, sz) += vals;
      } else {
        cart.slice(Sz3{c.x + st[0], c.y + st[1], c.z + st[2]}, sz) += vals;
      }
    }
  };

  auto const &start = log_.now();
  cart.setZero();
  Threads::RangeFor(grid_task, sortedIndices_.size());
  if (safe_) {
    log_.info("Combining thread workspaces");
    for (long ti = 0; ti < nThreads; ti++) {
      if (szZ[ti]) {
        cart.slice(Sz3{0, 0, minZ[ti]}, Sz3{cart.dimension(0), cart.dimension(1), szZ[ti]})
            .device(dev) += workspace[ti];
      }
    }
  }
  log_.debug("Non-cart -> Cart: {}", log_.toNow(start));
}

void Gridder::toCartesian(Cx3 const &noncart, Cx4 &cart) const
{
  assert(noncart.dimension(0) == info_.channels);
  assert(noncart.dimension(1) == info_.read_points);
  assert(noncart.dimension(2) == info_.spokes_total());
  assert(cart.dimension(0) == info_.channels);
  assert(cart.dimension(1) == dims_[0]);
  assert(cart.dimension(2) == dims_[1]);
  assert(cart.dimension(3) == dims_[2]);
  assert(sortedIndices_.size() == coords_.size());

  auto const st = kernel_->start();
  auto const sz = kernel_->size();
  auto dev = Threads::GlobalDevice();
  long const nThreads = dev.numThreads();
  std::vector<Cx4> workspace(nThreads);
  std::vector<long> minZ(nThreads, 0L), szZ(nThreads, 0L);
  auto grid_task = [&](long const lo, long const hi, long const ti) {
    // Allocate working space for this thread
    minZ[ti] = coords_[sortedIndices_[lo]].cart.z - kernel_->radius();

    if (safe_) {
      long const maxZ = coords_[sortedIndices_[hi - 1]].cart.z + kernel_->radius() + 1;
      szZ[ti] = maxZ - minZ[ti];
      workspace[ti].resize(cart.dimension(0), cart.dimension(1), cart.dimension(2), szZ[ti]);
      workspace[ti].setZero();
    }

    for (auto ii = lo; ii < hi; ii++) {
      if (lo == 0) {
        log_.progress(ii, hi);
      }
      auto const &cp = coords_[sortedIndices_[ii]];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      auto const &dc = pow(cp.sdc, DCexp_);
      Cx4 const nck = noncart.chip(nc.spoke, 2)
                          .chip(nc.read, 1)
                          .reshape(Sz4{info_.channels, 1, 1, 1})
                          .broadcast(Sz4{1, sz[0], sz[1], sz[2]});
      R4 const weights = Tile(kernel_->kspace(cp.offset), info_.channels) * dc;
      auto const vals = nck * weights.cast<Cx>();

      if (safe_) {
        workspace[ti].slice(
            Sz4{0, c.x + st[0], c.y + st[1], c.z - minZ[ti] + st[2]},
            Sz4{info_.channels, sz[0], sz[1], sz[2]}) += vals;
      } else {
        cart.slice(
            Sz4{0, c.x + st[0], c.y + st[1], c.z + st[2]},
            Sz4{info_.channels, sz[0], sz[1], sz[2]}) += vals;
      }
    }
  };

  auto const &start = log_.now();
  cart.setZero();
  Threads::RangeFor(grid_task, sortedIndices_.size());
  if (safe_) {
    log_.info("Combining thread workspaces");
    for (long ti = 0; ti < nThreads; ti++) {
      if (szZ[ti]) {
        cart.slice(
                Sz4{0, 0, 0, minZ[ti]},
                Sz4{cart.dimension(0), cart.dimension(1), cart.dimension(2), szZ[ti]})
            .device(dev) += workspace[ti];
      }
    }
  }
  log_.debug("Non-cart -> Cart: {}", log_.toNow(start));
}

void Gridder::toNoncartesian(Cx3 const &cart, Cx2 &noncart) const
{
  assert(noncart.dimension(0) == info_.read_points);
  assert(noncart.dimension(1) == info_.spokes_total());
  assert(cart.dimension(0) == dims_[0]);
  assert(cart.dimension(1) == dims_[1]);
  assert(cart.dimension(2) == dims_[2]);

  auto const st = kernel_->start();
  auto const sz = kernel_->size();
  Log nullLog;
  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      if (lo == 0) {
        log_.progress(ii, hi);
      }
      // for (auto ii = 0; ii < coords_.size(); ii++) {
      auto const &cp = coords_[ii];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      auto const &ksl = cart.slice(Sz3{c.x + st[0], c.y + st[1], c.z + st[2]}, sz);
      Cx0 const val = ksl.contract(
          kernel_->kspace(cp.offset).cast<Cx>(),
          Eigen::IndexPairList<
              Eigen::type2indexpair<0, 0>,
              Eigen::type2indexpair<1, 1>,
              Eigen::type2indexpair<2, 2>>());
      noncart.chip(nc.spoke, 1).chip(nc.read, 0) = val;
    }
  };
  auto const &start = log_.now();
  Threads::RangeFor(grid_task, coords_.size());
  log_.debug("Cart -> Non-cart: {}", log_.toNow(start));
}

void Gridder::toNoncartesian(Cx4 const &cart, Cx3 &noncart) const
{
  assert(noncart.dimension(0) == info_.channels);
  assert(noncart.dimension(1) == info_.read_points);
  assert(noncart.dimension(2) == info_.spokes_total());
  assert(cart.dimension(0) == info_.channels);
  assert(cart.dimension(1) == dims_[0]);
  assert(cart.dimension(2) == dims_[1]);
  assert(cart.dimension(3) == dims_[2]);

  auto const st = kernel_->start();
  auto const sz = kernel_->size();
  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      if (lo == 0) {
        log_.progress(ii, hi);
      }
      auto const &cp = coords_[ii];
      auto const &c = cp.cart;
      auto const &nc = cp.noncart;
      auto const &ksl = cart.slice(
          Sz4{0, c.x + st[0], c.y + st[1], c.z + st[2]}, Sz4{info_.channels, sz[0], sz[1], sz[2]});
      R3 const w = kernel_->kspace(cp.offset);
      noncart.chip(nc.spoke, 2).chip(nc.read, 1) = ksl.contract(
          w.cast<Cx>(),
          Eigen::IndexPairList<
              Eigen::type2indexpair<1, 0>,
              Eigen::type2indexpair<2, 1>,
              Eigen::type2indexpair<3, 2>>());
    }
  };
  auto const &start = log_.now();
  Threads::RangeFor(grid_task, coords_.size());
  log_.debug("Cart -> Non-cart: {}", log_.toNow(start));
}
