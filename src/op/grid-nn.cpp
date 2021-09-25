#include "grid-nn.h"

#include "../tensorOps.h"
#include "../threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

GridNN::GridNN(
    Trajectory const &traj,
    float const os,
    bool const unsafe,
    Log &log,
    float const inRes,
    bool const shrink)
    : GridOp(traj.mapping(os, 0, inRes, shrink), unsafe, log)
{
}

void GridNN::Adj(Cx3 const &noncart, Cx4 &cart) const
{
  assert(noncart.dimension(0) == cart.dimension(0));
  assert(cart.dimension(1) == mapping_.cartDims[0]);
  assert(cart.dimension(2) == mapping_.cartDims[1]);
  assert(cart.dimension(3) == mapping_.cartDims[2]);
  assert(mapping_.sortedIndices.size() == mapping_.cart.size());

  auto dev = Threads::GlobalDevice();
  long const nThreads = dev.numThreads();
  std::vector<Cx4> workspace(nThreads);
  std::vector<long> minZ(nThreads, 0L), szZ(nThreads, 0L);
  auto grid_task = [&](long const lo, long const hi, long const ti) {
    // Allocate working space for this thread
    minZ[ti] = mapping_.cart[mapping_.sortedIndices[lo]].z;

    if (safe_) {
      long const maxZ = mapping_.cart[mapping_.sortedIndices[hi - 1]].z;
      szZ[ti] = maxZ - minZ[ti] + 1;
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
      if (safe_) {
        workspace[ti].chip(c.z - minZ[ti], 3).chip(c.y, 2).chip(c.x, 1) +=
            noncart.chip(nc.spoke, 2).chip(nc.read, 1) *
            noncart.chip(nc.spoke, 2).chip(nc.read, 1).constant(dc);
      } else {
        cart.chip(c.z, 3).chip(c.y, 2).chip(c.x, 1) +=
            noncart.chip(nc.spoke, 2).chip(nc.read, 1) *
            noncart.chip(nc.spoke, 2).chip(nc.read, 1).constant(dc);
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

void GridNN::A(Cx4 const &cart, Cx3 &noncart) const
{
  assert(noncart.dimension(0) == cart.dimension(0));
  assert(cart.dimension(1) == mapping_.cartDims[0]);
  assert(cart.dimension(2) == mapping_.cartDims[1]);
  assert(cart.dimension(3) == mapping_.cartDims[2]);

  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      log_.progress(ii, lo, hi);
      auto const si = mapping_.sortedIndices[ii];
      auto const c = mapping_.cart[si];
      auto const nc = mapping_.noncart[si];
      auto const offset = mapping_.offset[si];
      noncart.chip(nc.spoke, 2).chip(nc.read, 1) = cart.chip(c.z, 3).chip(c.y, 2).chip(c.x, 1);
    }
  };
  auto const &start = log_.now();
  noncart.setZero();
  Threads::RangeFor(grid_task, mapping_.cart.size());
  log_.debug("Cart -> Non-cart: {}", log_.toNow(start));
}

R3 GridNN::apodization(Sz3 const sz) const
{
  R3 a(sz);
  a.setConstant(1.f);
  return a;
}
