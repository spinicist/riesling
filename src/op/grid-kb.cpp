#include "grid-kb.h"

#include "../cropper.h"
#include "../fft_plan.h"
#include "../tensorOps.h"
#include "../threads.h"

#include <algorithm>
#include <cfenv>
#include <cmath>

template <int InPlane, int ThroughPlane>
GridKB<InPlane, ThroughPlane>::GridKB(
    Trajectory const &traj,
    float const os,
    bool const unsafe,
    Log &log,
    float const inRes,
    bool const shrink)
    : GridOp(traj.mapping(os, (InPlane / 2), inRes, shrink), unsafe, log)
    , betaIn_{(float)M_PI * sqrtf(pow(InPlane * (mapping_.osamp - 0.5f) / mapping_.osamp, 2.f) - 0.8f)}
    , betaThrough_{(float)M_PI * sqrtf(pow(ThroughPlane * (mapping_.osamp - 0.5f) / mapping_.osamp, 2.f) - 0.8f)}
    , fft_{Sz3{InPlane, InPlane, ThroughPlane}, Log(), 1}
{

  // Array of indices used when building the kernel
  std::iota(indIn_.data(), indIn_.data() + InPlane, -InPlane / 2);
  std::iota(indThrough_.data(), indThrough_.data() + ThroughPlane, -ThroughPlane / 2);
}

template <int W, typename T>
inline decltype(auto) KB(T const &x, float const beta)
{
  return (x > (W / 2.f))
      .select(
          x.constant(0.f),
          (x.constant(beta) * (x.constant(1.f) - (x * x.constant(2.f / W)).square()).sqrt())
              .bessel_i0());
}

template <int InPlane, int ThroughPlane>
void GridKB<InPlane, ThroughPlane>::kernel(Point3 const r, float const dc, Kernel &k) const
{
  InPlaneArray const kx = KB<InPlane>(indIn_.constant(r[0]) - indIn_, betaIn_);
  InPlaneArray const ky = KB<InPlane>(indIn_.constant(r[2]) - indIn_, betaIn_);

  if constexpr (ThroughPlane > 1) {
    ThroughPlaneArray const kz =
        KB<ThroughPlane>(indThrough_.constant(r[3]) - indThrough_, betaThrough_);
    k = Outer(Outer(kx, ky), kz);
  } else {
    k = Outer(kx, ky);
  }

  if (sqrt_) {
    // This is the worst possible way to do this but I cannot figure out what IFFT(SQRT(FFT(KB))) is
    Cx3 temp(InPlane, InPlane, ThroughPlane);
    temp = k.template cast<Cx>();
    fft_.reverse(temp);
    temp.sqrt();
    fft_.forward(temp);
    k = temp.real();
  }
  k = k * dc / Sum(k);
}

template <int InPlane, int ThroughPlane>
void GridKB<InPlane, ThroughPlane>::Adj(Cx3 const &noncart, Cx4 &cart) const
{
  assert(noncart.dimension(0) == cart.dimension(0));
  assert(cart.dimension(1) == mapping_.cartDims[0]);
  assert(cart.dimension(2) == mapping_.cartDims[1]);
  assert(cart.dimension(3) == mapping_.cartDims[2]);
  assert(mapping_.sortedIndices.size() == mapping_.cart.size());

  long const nchan = cart.dimension(0);
  using FixZero = Eigen::type2index<0>;
  using FixOne = Eigen::type2index<1>;
  using FixIn = Eigen::type2index<InPlane>;
  using FixThrough = Eigen::type2index<ThroughPlane>;
  Eigen::IndexList<int, FixOne, FixOne, FixOne> rshNC;
  Eigen::IndexList<FixOne, FixIn, FixIn, FixThrough> brdNC;
  rshNC.set(0, nchan);

  Eigen::IndexList<FixOne, FixIn, FixIn, FixThrough> rshC;
  Eigen::IndexList<int, FixOne, FixOne, FixOne> brdC;
  brdC.set(0, nchan);

  Eigen::IndexList<int, FixIn, FixIn, FixThrough> szC;
  szC.set(0, nchan);

  auto dev = Threads::GlobalDevice();
  long const nThreads = dev.numThreads();
  std::vector<Cx4> workspace(nThreads);
  std::vector<long> minZ(nThreads, 0L), szZ(nThreads, 0L);
  auto grid_task = [&](long const lo, long const hi, long const ti) {
    // Allocate working space for this thread
    Kernel k;
    Eigen::IndexList<FixZero, int, int, int> stC;
    minZ[ti] = mapping_.cart[mapping_.sortedIndices[lo]].z - ((ThroughPlane - 1) / 2);

    if (safe_) {
      long const maxZ = mapping_.cart[mapping_.sortedIndices[hi - 1]].z + (ThroughPlane / 2);
      szZ[ti] = maxZ - minZ[ti] + 1;
      workspace[ti].resize(cart.dimension(0), cart.dimension(1), cart.dimension(2), szZ[ti]);
      workspace[ti].setZero();
    }

    for (auto ii = lo; ii < hi; ii++) {
      log_.progress(ii, lo, hi);
      auto const si = mapping_.sortedIndices[ii];
      auto const c = mapping_.cart[si];
      auto const nc = mapping_.noncart[si];
      auto const nck = noncart.chip(nc.spoke, 2).chip(nc.read, 1);
      kernel(mapping_.offset[si], pow(mapping_.sdc[si], DCexp_), k);
      stC.set(1, c.x - (InPlane / 2));
      stC.set(2, c.y - (InPlane / 2));
      if (safe_) {
        stC.set(3, c.z - (ThroughPlane / 2) - minZ[ti]);
        workspace[ti].slice(stC, szC) += nck.reshape(rshNC).broadcast(brdNC) *
                                         k.template cast<Cx>().reshape(rshC).broadcast(brdC);
      } else {
        stC.set(3, c.z - (ThroughPlane / 2));
        cart.slice(stC, szC) += nck.reshape(rshNC).broadcast(brdNC) *
                                k.template cast<Cx>().reshape(rshC).broadcast(brdC);
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

template <int InPlane, int ThroughPlane>
void GridKB<InPlane, ThroughPlane>::A(Cx4 const &cart, Cx3 &noncart) const
{
  assert(noncart.dimension(0) == cart.dimension(0));
  assert(cart.dimension(1) == mapping_.cartDims[0]);
  assert(cart.dimension(2) == mapping_.cartDims[1]);
  assert(cart.dimension(3) == mapping_.cartDims[2]);

  long const nchan = cart.dimension(0);
  using FixZero = Eigen::type2index<0>;
  using FixIn = Eigen::type2index<InPlane>;
  using FixThrough = Eigen::type2index<ThroughPlane>;
  Eigen::IndexList<int, FixIn, FixIn, FixThrough> szC;
  szC.set(0, nchan);

  auto grid_task = [&](long const lo, long const hi) {
    Kernel k;
    Eigen::IndexList<FixZero, int, int, int> stC;
    for (auto ii = lo; ii < hi; ii++) {
      log_.progress(ii, lo, hi);
      auto const si = mapping_.sortedIndices[ii];
      auto const c = mapping_.cart[si];
      auto const nc = mapping_.noncart[si];
      kernel(mapping_.offset[si], 1.f, k);
      stC.set(1, c.x - (InPlane / 2));
      stC.set(2, c.y - (InPlane / 2));
      stC.set(3, c.z - (ThroughPlane / 2));
      noncart.chip(nc.spoke, 2).chip(nc.read, 1) = cart.slice(stC, szC).contract(
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
R3 GridKB<InPlane, ThroughPlane>::apodization(Sz3 const sz) const
{
  // There is an analytic expression for this but I haven't got it right to date
  auto gridSz = this->gridDims();
  Cx3 temp(gridSz);
  FFT::ThreeD fft(temp, log_);
  temp.setZero();
  Kernel k;
  kernel(Point3{0, 0, 0}, 1.f, k);
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

template struct GridKB<3, 3>;
template struct GridKB<3, 1>;
