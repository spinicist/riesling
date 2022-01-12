#pragma once

#include "grid-basis.hpp"

#include "../tensorOps.h"
#include "../threads.h"
#include "../trajectory.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

struct GridBasisNN final : GridBasisOp
{
  GridBasisNN(
    Trajectory const &traj,
    float const os,
    bool const unsafe,
    R2 const &basis,
    Log &log,
    float const inRes = -1.f,
    bool const shrink = false)
    : GridBasisOp(traj.mapping(os, 0, inRes, shrink), unsafe, basis, log)
  {
  }

  GridBasisNN(Mapping const &map, bool const unsafe, R2 const &basis, Log &log)
    : GridBasisOp(map, unsafe, basis, log)
  {
  }

  void Adj(Output const &noncart, Input &cart) const
  {
    assert(cart.dimension(0) == noncart.dimension(0));
    assert(cart.dimension(1) == basis_.dimension(1));
    assert(cart.dimension(2) == mapping_.cartDims[0]);
    assert(cart.dimension(3) == mapping_.cartDims[1]);
    assert(cart.dimension(4) == mapping_.cartDims[2]);
    assert(mapping_.sortedIndices.size() == mapping_.cart.size());

    Index const nchan = cart.dimension(0);
    Index const nB = cart.dimension(1);

    Eigen::IndexList<int, FixOne> rshNC;
    Eigen::IndexList<FixOne, int> brdNC;
    rshNC.set(0, nchan);
    brdNC.set(1, nB);

    Eigen::IndexList<FixOne, int> rshB;
    Eigen::IndexList<int, FixOne> brdB;
    rshB.set(1, nB);
    brdB.set(0, nchan);

    auto dev = Threads::GlobalDevice();
    Index const nThreads = dev.numThreads();
    std::vector<Cx5> workspace(nThreads);
    std::vector<Index> minZ(nThreads, 0L), szZ(nThreads, 0L);
    auto grid_task = [&](Index const lo, Index const hi, Index const ti) {
      // Allocate working space for this thread
      minZ[ti] = mapping_.cart[mapping_.sortedIndices[lo]].z;

      if (safe_) {
        Index const maxZ = mapping_.cart[mapping_.sortedIndices[hi - 1]].z;
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
        auto const sdc = pow(mapping_.sdc[si], sdcPow_);
        auto const b =
          (basis_.chip<0>(nc.spoke % basis_.dimension(0)) * sdc * basisScale_).cast<Cx>();
        if (safe_) {
          workspace[ti].chip<4>(c.z - minZ[ti]).chip<3>(c.y).chip<2>(c.x) +=
            noncart.chip<2>(nc.spoke).chip<1>(nc.read).reshape(rshNC).broadcast(brdNC) *
            b.reshape(rshB).broadcast(brdB);
        } else {
          cart.chip<4>(c.z).chip<3>(c.y).chip<2>(c.x) +=
            noncart.chip<2>(nc.spoke).chip<1>(nc.read).reshape(rshNC).broadcast(brdNC) *
            b.reshape(rshB).broadcast(brdB);
        }
      }
    };

    auto const &start = log_.now();
    cart.setZero();
    Threads::RangeFor(grid_task, mapping_.cart.size());
    if (safe_) {
      log_.info("Combining thread workspaces...");
      for (Index ti = 0; ti < nThreads; ti++) {
        if (szZ[ti]) {
          cart
            .slice(
              Sz5{0, 0, 0, 0, minZ[ti]},
              Sz5{
                cart.dimension(0),
                cart.dimension(1),
                cart.dimension(2),
                cart.dimension(3),
                szZ[ti]})
            .device(dev) += workspace[ti];
        }
      }
    }
    log_.debug("Non-cart -> Cart: {}", log_.toNow(start));
  }

  void A(Input const &cart, Output &noncart) const
  {
    assert(cart.dimension(0) == noncart.dimension(0));
    assert(cart.dimension(1) == basis_.dimension(1));
    assert(cart.dimension(2) == mapping_.cartDims[0]);
    assert(cart.dimension(3) == mapping_.cartDims[1]);
    assert(cart.dimension(4) == mapping_.cartDims[2]);

    auto grid_task = [&](Index const lo, Index const hi) {
      for (auto ii = lo; ii < hi; ii++) {
        log_.progress(ii, lo, hi);
        auto const si = mapping_.sortedIndices[ii];
        auto const c = mapping_.cart[si];
        auto const nc = mapping_.noncart[si];
        noncart.chip<2>(nc.spoke).chip<1>(nc.read) =
          cart.chip<4>(c.z).chip<3>(c.y).chip<2>(c.x).contract(
            basis_.chip<0>(nc.spoke % basis_.dimension(0)).cast<Cx>(),
            Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>()) *
          noncart.chip<2>(nc.spoke).chip<1>(nc.read).constant(basisScale_);
      }
    };
    auto const &start = log_.now();
    noncart.setZero();
    Threads::RangeFor(grid_task, mapping_.cart.size());
    log_.debug("Cart -> Non-cart: {}", log_.toNow(start));
  }

  R3 apodization(Sz3 const sz) const
  {
    R3 a(sz);
    a.setConstant(1.f);
    return a;
  }
};
