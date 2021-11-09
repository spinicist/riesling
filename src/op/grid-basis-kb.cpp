#include "grid-basis-kb.h"

#include "../cropper.h"
#include "../fft_plan.h"
#include "../tensorOps.h"
#include "../threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

template <int InPlane, int ThroughPlane>
GridBasisKB<InPlane, ThroughPlane>::GridBasisKB(
  Trajectory const &traj,
  float const os,
  bool const unsafe,
  R2 const &basis,
  Log &log,
  float const inRes,
  bool const shrink)
  : GridBasisOp(traj.mapping(os, (InPlane / 2), inRes, shrink), unsafe, basis, log)
  , kernel_{os}
{
}

template <int InPlane, int ThroughPlane>
GridBasisKB<InPlane, ThroughPlane>::GridBasisKB(
  Mapping const &map, bool const unsafe, R2 const &basis, Log &log)
  : GridBasisOp(map, unsafe, basis, log)
  , kernel_{mapping_.osamp}
{
}

template <int InPlane, int ThroughPlane>
void GridBasisKB<InPlane, ThroughPlane>::sqrtOn()
{
  kernel_.sqrtOn();
}

template <int InPlane, int ThroughPlane>
void GridBasisKB<InPlane, ThroughPlane>::sqrtOff()
{
  kernel_.sqrtOff();
}

template <int InPlane, int ThroughPlane>
void GridBasisKB<InPlane, ThroughPlane>::Adj(Output const &noncart, Input &cart) const
{
  assert(cart.dimension(0) == noncart.dimension(0));
  assert(cart.dimension(1) == basis_.dimension(1));
  assert(cart.dimension(2) == mapping_.cartDims[0]);
  assert(cart.dimension(3) == mapping_.cartDims[1]);
  assert(cart.dimension(4) == mapping_.cartDims[2]);
  assert(mapping_.sortedIndices.size() == mapping_.cart.size());

  long const nchan = cart.dimension(0);
  long const nB = cart.dimension(1);

  Eigen::IndexList<int, FixOne, FixOne, FixOne, FixOne> rshNC;
  Eigen::IndexList<FixOne, int, FixIn, FixIn, FixThrough> brdNC;
  rshNC.set(0, nchan);
  brdNC.set(1, nB);

  Eigen::IndexList<FixOne, FixOne, FixIn, FixIn, FixThrough> rshK;
  Eigen::IndexList<int, int, FixOne, FixOne, FixOne> brdK;
  brdK.set(0, nchan);
  brdK.set(1, nB);

  Eigen::IndexList<FixOne, int, FixOne, FixOne, FixOne> rshB;
  Eigen::IndexList<int, FixOne, FixIn, FixIn, FixThrough> brdB;
  rshB.set(1, nB);
  brdB.set(0, nchan);

  Eigen::IndexList<int, int, FixIn, FixIn, FixThrough> szC;
  szC.set(0, nchan);
  szC.set(1, nB);

  auto dev = Threads::GlobalDevice();
  long const nThreads = dev.numThreads();
  std::vector<Cx5> workspace(nThreads);
  std::vector<long> minZ(nThreads, 0L), szZ(nThreads, 0L);
  auto grid_task = [&](long const lo, long const hi, long const ti) {
    // Allocate working space for this thread
    Eigen::IndexList<FixZero, FixZero, int, int, int> stC;
    minZ[ti] = mapping_.cart[mapping_.sortedIndices[lo]].z - ((ThroughPlane - 1) / 2);

    if (safe_) {
      long const maxZ = mapping_.cart[mapping_.sortedIndices[hi - 1]].z + (ThroughPlane / 2);
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
      auto const nck = noncart.chip(nc.spoke, 2).chip(nc.read, 1);
      auto const k = kernel_(mapping_.offset[si], mapping_.sdc[si] * basisScale_);
      stC.set(2, c.x - (InPlane / 2));
      stC.set(3, c.y - (InPlane / 2));
      if (safe_) {
        stC.set(4, c.z - (ThroughPlane / 2) - minZ[ti]);
        workspace[ti].slice(stC, szC) += nck.reshape(rshNC).broadcast(brdNC) *
                                         k.template cast<Cx>().reshape(rshK).broadcast(brdK) *
                                         basis_.chip(nc.spoke % basis_.dimension(0), 0)
                                           .template cast<Cx>()
                                           .reshape(rshB)
                                           .broadcast(brdB);
      } else {
        stC.set(4, c.z - (ThroughPlane / 2));
        cart.slice(stC, szC) += nck.reshape(rshNC).broadcast(brdNC) *
                                k.template cast<Cx>().reshape(rshK).broadcast(brdK) *
                                basis_.chip(nc.spoke % basis_.dimension(0), 0)
                                  .template cast<Cx>()
                                  .reshape(rshB)
                                  .broadcast(brdB);
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
        cart
          .slice(
            Sz5{0, 0, 0, 0, minZ[ti]},
            Sz5{
              cart.dimension(0), cart.dimension(1), cart.dimension(2), cart.dimension(3), szZ[ti]})
          .device(dev) += workspace[ti];
      }
    }
  }
  log_.debug("Non-cart -> Cart: {}", log_.toNow(start));
}

template <int InPlane, int ThroughPlane>
void GridBasisKB<InPlane, ThroughPlane>::A(Input const &cart, Output &noncart) const
{
  assert(cart.dimension(0) == noncart.dimension(0));
  assert(cart.dimension(1) == basis_.dimension(1));
  assert(cart.dimension(2) == mapping_.cartDims[0]);
  assert(cart.dimension(3) == mapping_.cartDims[1]);
  assert(cart.dimension(4) == mapping_.cartDims[2]);

  long const nchan = cart.dimension(0);
  long const nB = basis_.dimension(1);
  Eigen::IndexList<int, int, FixIn, FixIn, FixThrough> szC;
  szC.set(0, nchan);
  szC.set(1, nB);

  auto grid_task = [&](long const lo, long const hi) {
    Eigen::IndexList<FixZero, FixZero, int, int, int> stC;
    for (auto ii = lo; ii < hi; ii++) {
      log_.progress(ii, lo, hi);
      auto const si = mapping_.sortedIndices[ii];
      auto const c = mapping_.cart[si];
      auto const nc = mapping_.noncart[si];
      auto const k = kernel_(mapping_.offset[si], basisScale_);
      stC.set(2, c.x - (InPlane / 2));
      stC.set(3, c.y - (InPlane / 2));
      stC.set(4, c.z - (ThroughPlane / 2));
      noncart.chip(nc.spoke, 2).chip(nc.read, 1) =
        cart.slice(stC, szC)
          .contract(
            basis_.chip(nc.spoke % basis_.dimension(0), 0).template cast<Cx>(),
            Eigen::IndexPairList<Eigen::type2indexpair<1, 0>>())
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

template <int InPlane, int ThroughPlane>
R3 GridBasisKB<InPlane, ThroughPlane>::apodization(Sz3 const sz) const
{
  auto gridSz = this->gridDims();
  Cx3 temp(gridSz);
  FFT::ThreeD fft(temp, log_);
  temp.setZero();
  auto const k = kernel_(Point3{0, 0, 0}, 1.f);
  Crop3(temp, k.dimensions()) = k.template cast<Cx>();
  fft.reverse(temp);
  R3 a = Crop3(R3(temp.real()), sz);
  float const scale =
    sqrt(std::accumulate(gridSz.cbegin(), gridSz.cend(), 1, std::multiplies<long>()));
  log_.info(
    FMT_STRING("Apodization size {} scale factor: {}"), fmt::join(a.dimensions(), ","), scale);
  a.device(Threads::GlobalDevice()) = a * a.constant(scale);
  return a;
}

template struct GridBasisKB<3, 3>;
template struct GridBasisKB<3, 1>;
