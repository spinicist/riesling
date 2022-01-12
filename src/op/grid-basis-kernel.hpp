#pragma once

#include "grid-basis.hpp"

#include "../cropper.h"
#include "../kernel.h"
#include "../trajectory.h"

template <typename Kernel>
struct GridBasis final : GridBasisOp
{
  GridBasis(
    Trajectory const &traj,
    float const os,
    bool const unsafe,
    R2 const &basis,
    Log &log,
    float const inRes = -1.f,
    bool const shrink = false)
    : GridBasisOp(traj.mapping(os, (Kernel::InPlane / 2), inRes, shrink), unsafe, basis, log)
    , kernel_{os}
  {
  }

  GridBasis(Mapping const &mapping, bool const unsafe, R2 const &basis, Log &log)
    : GridBasisOp(mapping, unsafe, basis, log)
    , kernel_{mapping_.osamp}
  {
  }

  void A(Input const &cart, Output &noncart) const
  {
    assert(cart.dimension(0) == noncart.dimension(0));
    assert(cart.dimension(1) == basis_.dimension(1));
    assert(cart.dimension(2) == mapping_.cartDims[0]);
    assert(cart.dimension(3) == mapping_.cartDims[1]);
    assert(cart.dimension(4) == mapping_.cartDims[2]);

    Index const nchan = cart.dimension(0);
    Index const nB = basis_.dimension(1);
    Eigen::IndexList<int, int, FixIn, FixIn, FixThrough> szC;
    szC.set(0, nchan);
    szC.set(1, nB);

    auto grid_task = [&](Index const lo, Index const hi) {
      Eigen::IndexList<FixZero, FixZero, int, int, int> stC;
      for (auto ii = lo; ii < hi; ii++) {
        log_.progress(ii, lo, hi);
        auto const si = mapping_.sortedIndices[ii];
        auto const c = mapping_.cart[si];
        auto const n = mapping_.noncart[si];
        auto const b =
          (basis_.chip<0>(n.spoke % basis_.dimension(0)) * basisScale_).template cast<Cx>();
        auto const k = kernel_(mapping_.offset[si]);
        stC.set(2, c.x - (Kernel::InPlane / 2));
        stC.set(3, c.y - (Kernel::InPlane / 2));
        stC.set(4, c.z - (Kernel::ThroughPlane / 2));
        noncart.template chip<2>(n.spoke).template chip<1>(n.read) =
          cart.slice(stC, szC)
            .contract(b, Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>())
            .contract(
              k.template cast<Cx>(),
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

    constexpr Eigen::IndexList<FixOne, FixOne, FixIn, FixIn, FixThrough> rshK;
    Eigen::IndexList<int, int, FixOne, FixOne, FixOne> brdK;
    brdK.set(0, nchan);
    brdK.set(1, nB);

    Eigen::IndexList<int, int, FixOne, FixOne, FixOne> rshNCB;
    constexpr Eigen::IndexList<FixOne, FixOne, FixIn, FixIn, FixThrough> brdNCB;
    rshNCB.set(0, nchan);
    rshNCB.set(1, nB);

    Eigen::IndexList<int, int, FixIn, FixIn, FixThrough> szC;
    szC.set(0, nchan);
    szC.set(1, nB);

    auto dev = Threads::GlobalDevice();
    Index const nThreads = dev.numThreads();
    std::vector<Cx5> workspace(nThreads);
    std::vector<Index> minZ(nThreads, 0L), szZ(nThreads, 0L);
    auto grid_task = [&](Index const lo, Index const hi, Index const ti) {
      // Allocate working space for this thread
      Eigen::IndexList<FixZero, FixZero, int, int, int> stC;
      Cx2 ncb(nchan, nB);

      minZ[ti] = mapping_.cart[mapping_.sortedIndices[lo]].z - ((Kernel::ThroughPlane - 1) / 2);

      if (safe_) {
        Index const maxZ =
          mapping_.cart[mapping_.sortedIndices[hi - 1]].z + (Kernel::ThroughPlane / 2);
        szZ[ti] = maxZ - minZ[ti] + 1;
        workspace[ti].resize(
          cart.dimension(0), cart.dimension(1), cart.dimension(2), cart.dimension(3), szZ[ti]);
        workspace[ti].setZero();
      }

      for (auto ii = lo; ii < hi; ii++) {
        log_.progress(ii, lo, hi);
        auto const si = mapping_.sortedIndices[ii];
        auto const c = mapping_.cart[si];
        auto const n = mapping_.noncart[si];
        auto const sdc = pow(mapping_.sdc[si], sdcPow_);
        auto const nc = noncart.template chip<2>(n.spoke).template chip<1>(n.read);
        auto const b = (basis_.chip<0>(n.spoke % basis_.dimension(0)) * basisScale_);
        auto const k = kernel_(mapping_.offset[si]);
        ncb = (nc * nc.constant(sdc)).reshape(rshNC).broadcast(brdNC) *
              b.template cast<Cx>().reshape(rshB).broadcast(brdB);
        auto const nbk = ncb.reshape(rshNCB).broadcast(brdNCB) *
                         k.template cast<Cx>().reshape(rshK).broadcast(brdK);

        stC.set(2, c.x - (Kernel::InPlane / 2));
        stC.set(3, c.y - (Kernel::InPlane / 2));
        if (safe_) {
          stC.set(4, c.z - (Kernel::ThroughPlane / 2) - minZ[ti]);
          workspace[ti].slice(stC, szC) += nbk;
        } else {
          stC.set(4, c.z - (Kernel::ThroughPlane / 2));
          cart.slice(stC, szC) += nbk;
        }
      }
    };

    auto const start = log_.now();
    cart.setZero();
    Threads::RangeFor(grid_task, mapping_.cart.size());
    log_.debug("Basis Non-cart -> Cart: {}", log_.toNow(start));
    if (safe_) {
      log_.info("Combining thread workspaces...");
      auto const start2 = log_.now();
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
      log_.debug("Combining took: {}", log_.toNow(start2));
    }
  }

  R3 apodization(Sz3 const sz) const
  {
    auto gridSz = this->mapping().cartDims;
    Cx3 temp(gridSz);
    FFT::ThreeD fft(temp, log_);
    temp.setZero();
    auto const k = kernel_(Point3{0, 0, 0});
    Crop3(temp, k.dimensions()) = k.template cast<Cx>();
    fft.reverse(temp);
    R3 a = Crop3(R3(temp.real()), sz);
    float const scale =
      sqrt(std::accumulate(gridSz.cbegin(), gridSz.cend(), 1, std::multiplies<Index>()));
    log_.info(
      FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(a.dimensions(), ","), scale);
    a.device(Threads::GlobalDevice()) = a * a.constant(scale);
    return a;
  }

private:
  using FixIn = Eigen::type2index<Kernel::InPlane>;
  using FixThrough = Eigen::type2index<Kernel::ThroughPlane>;

  Kernel kernel_;
};
