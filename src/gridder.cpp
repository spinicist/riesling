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

Gridder::Gridder(
    Trajectory const &traj, float const os, Kernel *const kernel, bool const unsafe, Log &log)
    : info_{traj.info()}
    , oversample_{os}
    , DCexp_{1.f}
    , kernel_{kernel}
    , safe_{!unsafe}
    , log_{log}
{

  // Work out grid dimensions for the given oversampling
  long const gridSz = fft_size(oversample_ * info_.matrix.maxCoeff());
  if (info_.type == Info::Type::ThreeDStack) {
    dims_ = {gridSz, gridSz, info_.matrix[2]};
  } else {
    dims_ = {gridSz, gridSz, gridSz};
  }
  genCoords(traj, (gridSz / 2) - 1);
  sortCoords();
}

void Gridder::genCoords(Trajectory const &traj, long const nomRad)
{
  long const totalSz = info_.read_points * info_.spokes_total();
  coords_.clear();
  coords_.reserve(totalSz);

  std::fesetround(FE_TONEAREST);
  Size3 const center(dims_[0] / 2, dims_[1] / 2, dims_[2] / 2);
  auto start = log_.now();
  for (int32_t is = 0; is < info_.spokes_total(); is++) {
    for (int16_t ir = info_.read_gap; ir < info_.read_points; ir++) {
      NoncartesianIndex const nc{.spoke = is, .read = ir};
      Point3 const xyz = traj.point(ir, is, nomRad);

      // Only grid lo-res to where hi-res begins (i.e. fill the dead-time gap)
      // Otherwise leave space for kernel
      float const maxRad = (is < info_.spokes_lo)
                               ? oversample_ * info_.read_gap / info_.spoke_oversamp()
                               : nomRad - kernel_->radius();
      if (xyz.norm() <= maxRad) {
        Size3 const gp = nearby(xyz).cast<int16_t>();
        Point3 const offset = xyz - gp.cast<float>().matrix();
        Size3 const cart = gp + center;
        coords_.push_back(Coords{.cart = CartesianIndex{cart(0), cart(1), cart(2)},
                                 .noncart = nc,
                                 .sdc = 1.f,
                                 .offset = offset});
      }
    }
  }
  log_.info("Generated {} co-ordinates in {}", coords_.size(), log_.toNow(start));
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

void Gridder::setSDC(float const d)
{
  for (auto &c : coords_) {
    c.sdc = d;
  }
}

void Gridder ::setSDC(R2 const &sdc)
{
  if (info_.read_points != sdc.dimension(0) || info_.spokes_total() != sdc.dimension(1)) {
    log_.fail("SDC dimensions {} do not match trajectory", fmt::join(sdc.dimensions(), ","));
  }
  for (auto &c : coords_) {
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

Info const &Gridder::info() const
{
  return info_;
}

float Gridder::oversample() const
{
  return oversample_;
}

Kernel *Gridder::kernel() const
{
  return kernel_;
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
