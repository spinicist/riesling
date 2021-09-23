#include "gridder.h"

#include "fft_plan.h"
#include "filter.h"
#include "tensorOps.h"
#include "threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

Gridder::Gridder(Mapping map, Kernel *const kernel, bool const unsafe, Log &log)
    : mapping_{std::move(map)}
    , kernel_{kernel}
    , safe_{!unsafe}
    , log_{log}
    , DCexp_{1.f}
{
}

Sz3 Gridder::gridDims() const
{
  return mapping_.cartDims;
}

Cx4 Gridder::newMultichannel(long const nc) const
{
  Cx4 g(nc, mapping_.cartDims[0], mapping_.cartDims[1], mapping_.cartDims[2]);
  g.setZero();
  return g;
}

void Gridder::setSDC(float const d)
{
  std::fill(mapping_.sdc.begin(), mapping_.sdc.end(), d);
}

void Gridder ::setSDC(R2 const &sdc)
{
  std::transform(
      mapping_.noncart.begin(),
      mapping_.noncart.end(),
      mapping_.sdc.begin(),
      [&sdc](NoncartesianIndex const &nc) { return sdc(nc.read, nc.spoke); });
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

Kernel *Gridder::kernel() const
{
  return kernel_;
}

void Gridder::toCartesian(Cx3 const &noncart, Cx4 &cart) const
{
  assert(noncart.dimension(0) == cart.dimension(0));
  assert(noncart.dimension(1) == info_.read_points);
  assert(noncart.dimension(2) == info_.spokes_total());
  assert(cart.dimension(1) == dims_[0]);
  assert(cart.dimension(2) == dims_[1]);
  assert(cart.dimension(3) == dims_[2]);
  assert(sortedIndices_.size() == coords_.size());

  long const nchan = cart.dimension(0);
  auto const st = kernel_->start();
  auto const sz = kernel_->size();
  auto dev = Threads::GlobalDevice();
  long const nThreads = dev.numThreads();
  std::vector<Cx4> workspace(nThreads);
  std::vector<long> minZ(nThreads, 0L), szZ(nThreads, 0L);
  auto grid_task = [&](long const lo, long const hi, long const ti) {
    // Allocate working space for this thread
    minZ[ti] = mapping_.cart[mapping_.sortedIndices[lo]].z - kernel_->radius();

    if (safe_) {
      long const maxZ = mapping_.cart[mapping_.sortedIndices[hi - 1]].z + kernel_->radius() + 1;
      szZ[ti] = maxZ - minZ[ti];
      workspace[ti].resize(cart.dimension(0), cart.dimension(1), cart.dimension(2), szZ[ti]);
      workspace[ti].setZero();
    }

    for (auto ii = lo; ii < hi; ii++) {
      log_.progress(ii, lo, hi);
      auto const si = mapping_.sortedIndices[ii];
      auto const c = mapping_.cart[si];
      auto const nc = mapping_.noncart[si];
      auto const dc = pow(mapping_.sdc[si], DCexp_);
      auto const offset = mapping_.offset[si];
      Cx4 const nck = noncart.chip(nc.spoke, 2)
                          .chip(nc.read, 1)
                          .reshape(Sz4{nchan, 1, 1, 1})
                          .broadcast(Sz4{1, sz[0], sz[1], sz[2]});
      R4 const weights = Tile(kernel_->kspace(offset), nchan) * dc;
      auto const vals = nck * weights.cast<Cx>();

      if (safe_) {
        workspace[ti].slice(
            Sz4{0, c.x + st[0], c.y + st[1], c.z - minZ[ti] + st[2]},
            Sz4{nchan, sz[0], sz[1], sz[2]}) += vals;
      } else {
        cart.slice(
            Sz4{0, c.x + st[0], c.y + st[1], c.z + st[2]}, Sz4{nchan, sz[0], sz[1], sz[2]}) += vals;
      }
    }
  };

  auto const &start = log_.now();
  cart.setZero();
  Threads::RangeFor(grid_task, mapping_.cart.size());
  if (safe_) {
    log_.info("Combining thread workspaces...");
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

void Gridder::toNoncartesian(Cx4 const &cart, Cx3 &noncart) const
{
  assert(noncart.dimension(0) == cart.dimension(0));
  assert(noncart.dimension(1) == info_.read_points);
  assert(noncart.dimension(2) == info_.spokes_total());
  assert(cart.dimension(1) == dims_[0]);
  assert(cart.dimension(2) == dims_[1]);
  assert(cart.dimension(3) == dims_[2]);

  long const nchan = cart.dimension(0);
  auto const st = kernel_->start();
  auto const sz = kernel_->size();
  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      log_.progress(ii, lo, hi);
      auto const si = mapping_.sortedIndices[ii];
      auto const c = mapping_.cart[si];
      auto const nc = mapping_.noncart[si];
      auto const offset = mapping_.offset[si];
      auto const &ksl = cart.slice(
          Sz4{0, c.x + st[0], c.y + st[1], c.z + st[2]}, Sz4{nchan, sz[0], sz[1], sz[2]});
      R3 const w = kernel_->kspace(offset);
      noncart.chip(nc.spoke, 2).chip(nc.read, 1) = ksl.contract(
          w.cast<Cx>(),
          Eigen::IndexPairList<
              Eigen::type2indexpair<1, 0>,
              Eigen::type2indexpair<2, 1>,
              Eigen::type2indexpair<3, 2>>());
    }
  };
  auto const &start = log_.now();
  noncart.setZero();
  Threads::RangeFor(grid_task, mapping_.cart.size());
  log_.debug("Cart -> Non-cart: {}", log_.toNow(start));
}
