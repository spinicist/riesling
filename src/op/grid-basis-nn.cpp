#include "grid-basis-nn.h"

#include "../tensorOps.h"
#include "../threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

GridBasisNN::GridBasisNN(
    Trajectory const &traj,
    float const os,
    bool const unsafe,
    R2 const &basis,
    Log &log,
    float const inRes,
    bool const shrink)
    : GridBasisOp(traj.mapping(os, 0, inRes, shrink), unsafe, basis, log)
{
}

GridBasisNN::GridBasisNN(
  Mapping const &map,
  bool const unsafe,
  R2 const &basis,
  Log &log)
  : GridBasisOp(map, unsafe, basis, log)
{}


void GridBasisNN::Adj(Output const &noncart, Input &cart) const
{
  assert(cart.dimension(0) == noncart.dimension(0));
  assert(cart.dimension(1) == basis_.dimension(1));
  assert(cart.dimension(2) == mapping_.cartDims[0]);
  assert(cart.dimension(3) == mapping_.cartDims[1]);
  assert(cart.dimension(4) == mapping_.cartDims[2]);
  assert(mapping_.sortedIndices.size() == mapping_.cart.size());

  using FixOne = Eigen::type2index<1>;
  long const nchan = cart.dimension(0);
  long const nB = cart.dimension(1);

  Eigen::IndexList<int, FixOne> rshNC;
  Eigen::IndexList<FixOne, int> brdNC;
  rshNC.set(0, nchan);
  brdNC.set(1, nB);

  Eigen::IndexList<FixOne, int> rshB;
  Eigen::IndexList<int, FixOne> brdB;
  rshB.set(1, nB);
  brdB.set(0, nchan);

  auto dev = Threads::GlobalDevice();
  long const nThreads = dev.numThreads();
  std::vector<Cx5> workspace(nThreads);
  std::vector<long> minZ(nThreads, 0L), szZ(nThreads, 0L);
  auto grid_task = [&](long const lo, long const hi, long const ti) {
    // Allocate working space for this thread
    minZ[ti] = mapping_.cart[mapping_.sortedIndices[lo]].z;

    if (safe_) {
      long const maxZ = mapping_.cart[mapping_.sortedIndices[hi - 1]].z;
      szZ[ti] = maxZ - minZ[ti] + 1;
      workspace[ti].resize(
          cart.dimension(0), cart.dimension(1), cart.dimension(2), cart.dimension(3), szZ[ti]);
      workspace[ti].setZero();
    }

    for (auto ii = lo; ii < hi; ii++) {
      log_.progress(ii, lo, hi);
      auto const si = mapping_.sortedIndices[ii];
      auto const c = mapping_.cart[si];
      auto const nc = mapping_.noncart[si];
      auto const dc = pow(mapping_.sdc[si], DCexp_);
      if (safe_) {
        workspace[ti].chip(c.z - minZ[ti], 4).chip(c.y, 3).chip(c.x, 2) +=
            (noncart.chip(nc.spoke, 2).chip(nc.read, 1) *
             noncart.chip(nc.spoke, 2).chip(nc.read, 1).constant(dc))
                .reshape(rshNC)
                .broadcast(brdNC) *
            basis_.chip(nc.spoke % basis_.dimension(0), 0).cast<Cx>().reshape(rshB).broadcast(brdB);
      } else {
        cart.chip(c.z, 4).chip(c.y, 3).chip(c.x, 2) +=
            (noncart.chip(nc.spoke, 2).chip(nc.read, 1) *
             noncart.chip(nc.spoke, 2).chip(nc.read, 1).constant(dc))
                .reshape(rshNC)
                .broadcast(brdNC) *
            basis_.chip(nc.spoke % basis_.dimension(0), 0).cast<Cx>().reshape(rshB).broadcast(brdB);
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
                Sz5{0, 0, 0, 0, minZ[ti]},
                Sz5{cart.dimension(0), cart.dimension(1), cart.dimension(2), cart.dimension(3), szZ[ti]})
            .device(dev) += workspace[ti];
      }
    }
  }
  log_.debug("Non-cart -> Cart: {}", log_.toNow(start));
}

void GridBasisNN::A(Input const &cart, Output &noncart) const
{
  assert(cart.dimension(0) == noncart.dimension(0));
  assert(cart.dimension(1) == basis_.dimension(1));
  assert(cart.dimension(2) == mapping_.cartDims[0]);
  assert(cart.dimension(3) == mapping_.cartDims[1]);
  assert(cart.dimension(4) == mapping_.cartDims[2]);

  auto grid_task = [&](long const lo, long const hi) {
    for (auto ii = lo; ii < hi; ii++) {
      log_.progress(ii, lo, hi);
      auto const si = mapping_.sortedIndices[ii];
      auto const c = mapping_.cart[si];
      auto const nc = mapping_.noncart[si];
      noncart.chip(nc.spoke, 2).chip(nc.read, 1) =
          cart.chip(c.z, 4).chip(c.y, 3).chip(c.x, 2).contract(
              basis_.chip(nc.spoke % basis_.dimension(0), 0).cast<Cx>(),
              Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>());
    }
  };
  auto const &start = log_.now();
  noncart.setZero();
  Threads::RangeFor(grid_task, mapping_.cart.size());
  log_.debug("Cart -> Non-cart: {}", log_.toNow(start));
}

R3 GridBasisNN::apodization(Sz3 const sz) const
{
  R3 a(sz);
  a.setConstant(1.f);
  return a;
}
