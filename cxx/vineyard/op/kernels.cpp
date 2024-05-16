#include "kernels.hpp"

namespace rl {

template <typename Sc, int ND, int NK>
Kernels<Sc, ND, NK>::Kernels(InDims const ish, Sz<NK> const d, Sz<NK> const w)
  : Parent("KernelsOp")
  , kDims_{d}
  , kW_{w}
{
  ishape = ish;
  kSz_ = ishape;
  oshape = AddFront(ishape, 0);
  Index nK = 1;
  for (Index ii = 0; ii < NK; ii++) {
    auto const D = kDims_[ii];
    if (ishape[D] < kW_[ii]) { Log::Fail("Kernel size is bigger than image size"); }
    nK *= (ishape[D] - kW_[ii] + 1);
    kSz_[D] = kW_[ii];
    oshape[D + 1] = kW_[ii];
  }
  oshape[0] = nK;
  Log::Debug("ishape {} oshape {} kDims {} kW {}", ishape, oshape, kDims_, kW_);
}

template <typename Sc, int ND, int NK> void Kernels<Sc, ND, NK>::forward(InCMap const &x, OutMap &y) const
{
  auto const time = this->startForward(x);
  Index      ik = 0;
  Sz<ND>     st;
  st.fill(0);
  std::function<void(Index)> dimLoop = [&](Index ii) {
    Index const D = kDims_[ii];
    Index const nKd = x.dimension(D) - kW_[ii] + 1;
    for (Index id = 0; id < nKd; id++) {
      st[D] = id;
      if (ii == 0) {
        y.template chip<0>(ik++) = x.slice(st, kSz_);
      } else {
        dimLoop(ii - 1);
      }
    }
  };
  dimLoop(NK - 1);
  assert(ik == y.dimension(0));
  this->finishForward(y, time);
}

template <typename Sc, int ND, int NK> void Kernels<Sc, ND, NK>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  Sz<NK>     cSz;
  Sz<ND>     cRsh, cBrd;
  cRsh.fill(1);
  cBrd = ishape;
  for (Index ii = 0; ii < NK; ii++) {
    Index const D = kDims_[ii];
    cSz[ii] = ishape[D];
    cRsh[D] = ishape[D];
    cBrd[D] = 1;
  }
  ReN<NK> count(cSz);
  count.setZero();
  x.setZero();
  Index  ik = 0;
  Sz<ND> xSt;
  Sz<NK> cSt;
  xSt.fill(0);
  cSt.fill(0);
  std::function<void(Index)> dimLoop = [&](Index ii) {
    Index const D = kDims_[ii];
    Index const nKd = x.dimension(D) - kW_[ii] + 1;
    for (Index id = 0; id < nKd; id++) {
      xSt[D] = id;
      cSt[ii] = id;
      if (ii == 0) {
        x.slice(xSt, kSz_) += y.template chip<0>(ik++);
        count.slice(cSt, kW_) += count.slice(cSt, kW_).constant(1.f);
      } else {
        dimLoop(ii - 1);
      }
    }
  };
  dimLoop(NK - 1);
  assert(ik == y.dimension(0));
  x /= count.reshape(cRsh).broadcast(cBrd).template cast<Cx>();
  this->finishAdjoint(x, time);
}

template struct Kernels<Cx, 5, 3>;

} // namespace rl