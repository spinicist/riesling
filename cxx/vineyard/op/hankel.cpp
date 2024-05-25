#include "hankel.hpp"

namespace rl::TOps {

template <typename Sc, int ND, int NK>
Hankel<Sc, ND, NK>::Hankel(InDims const ish, Sz<NK> const d, Sz<NK> const w, const bool virt)
  : Parent("HankelOp")
  , kDims_{d}
  , kW_{w}
  , virt_{virt}
{
  ishape = ish;
  kSz_ = ishape;
  oshape = AddFront(ishape, 0, virt_ ? 2 : 1);
  Index nK = 1;
  for (Index ii = 0; ii < NK; ii++) {
    auto const D = kDims_[ii];
    if (ishape[D] < kW_[ii]) { Log::Fail("Kernel size is bigger than image size"); }
    nK *= ishape[D];
    kSz_[D] = kW_[ii];
    oshape[D + 2] = kW_[ii];
  }
  oshape[0] = nK;
  Log::Print("Hankel ishape {} oshape {} kDims {} kW {}", ishape, oshape, kDims_, kW_);
}

template <typename Sc, int ND, int NK> void Hankel<Sc, ND, NK>::forward(InCMap const &x, OutMap &y) const
{
  auto const             time = this->startForward(x);
  Index                  ik = 0;
  Sz<ND>                 st, roll, stSym, szSym;
  Eigen::array<bool, ND> rev;
  st.fill(0);
  roll.fill(0);
  stSym.fill(0);
  rev.fill(false);
  szSym = ishape;
  for (Index ii = 0; ii < NK; ii++) {
    Index const D = kDims_[ii];
    stSym[D] = 1;
    szSym[D] -= 1;
    rev[D] = true;
  }

  std::function<void(Index)> dimLoop = [&](Index ii) {
    Index const D = kDims_[ii];
    Index const nKd = x.dimension(D) - 1; // -1 because we are dropping the -N/2 samples
    for (Index id = 0; id < nKd; id++) {
      roll[D] = id;
      if (ii == 0) {
        y.template chip<1>(0).template chip<0>(ik) = x.slice(stSym, szSym).roll(roll).slice(st, kSz_);
        if (virt_) {
          y.template chip<1>(1).template chip<0>(ik) = x.slice(stSym, szSym).reverse(rev).roll(roll).slice(st, kSz_).conjugate();
        }
        ik++;
      } else {
        dimLoop(ii - 1);
      }
    }
  };
  dimLoop(NK - 1);
  assert(ik == y.dimension(0));
  this->finishForward(y, time);
}

template <typename Sc, int ND, int NK> void Hankel<Sc, ND, NK>::adjoint(OutCMap const &y, InMap &x) const
{
  auto const time = this->startAdjoint(y);
  x.setZero();
  Index                  ik = 0;
  Sz<ND>                 xSt, roll, stSym, szSym;
  Eigen::array<bool, ND> rev;
  xSt.fill(0);
  roll.fill(0);
  stSym.fill(0);
  rev.fill(false);
  szSym = ishape;
  for (Index ii = 0; ii < NK; ii++) {
    Index const D = kDims_[ii];
    stSym[D] = 1;
    szSym[D] -= 1;
    rev[D] = true;
  }

  std::function<void(Index)> dimLoop = [&](Index ii) {
    Index const D = kDims_[ii];
    Index const nKd = x.dimension(D) - 1;
    for (Index id = 0; id < nKd; id++) {
      roll[D] = id;
      if (ii == 0) {
        x.slice(stSym, szSym).roll(roll).slice(xSt, kSz_) += y.template chip<1>(0).template chip<0>(ik);
        if (virt_) {
          x.slice(stSym, szSym).reverse(rev).roll(roll).slice(xSt, kSz_) += y.template chip<1>(1).template chip<0>(ik).conjugate();
        }
        ik++;
      } else {
        dimLoop(ii - 1);
      }
    }
  };
  dimLoop(NK - 1);
  assert(ik == y.dimension(0));
  x.slice(stSym, szSym) /= x.slice(stSym, szSym).constant(Product(kW_) * (virt_ ? 2.f : 1.f));
  this->finishAdjoint(x, time);
}

template struct Hankel<Cx, 5, 3>;
template struct Hankel<Cx, 5, 1>;

} // namespace rl::TOps
