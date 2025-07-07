#include "hankel.hpp"

#include "../log/log.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

template <int ND, int NK> Hankel<ND, NK>::Hankel(InDims const ish, Sz<NK> const d, Sz<NK> const w, bool const sph, bool const v)
  : Parent("HankelOp")
  , kDims_{d}
  , kW_{w}
  , sphere_{sph}
  , virt_{v}
{
  ishape = ish;
  kSz_ = ishape;
  oshape = AddFront(ishape, 0, virt_ ? 2 : 1);
  Index                      nK = 1;
  Eigen::Array<float, NK, 1> rad, ind;
  for (Index ii = 0; ii < NK; ii++) {
    auto const D = kDims_[ii];
    if (ishape[D] < kW_[ii]) { throw Log::Failure("Hankel", "Kernel size is bigger than image size"); }
    kSz_[D] = kW_[ii];
    oshape[D + 2] = kW_[ii];
    if (!sphere_) {
      nK *= ishape[D];
    } else {
      rad[ii] = ishape[D] / 2;
    }
  }

  if (sphere_) {
    nK = 0;
    std::function<void(Index, Eigen::Array<float, NK, 1>)> dimLoop = [&](Index ii, Eigen::Array<float, NK, 1> ind) {
      Index const D = kDims_[ii];
      for (int id = 0; id < ishape[D] - 1; id++) {
        ind[ii] = id;
        if (ii == 0) {
          if (((ind / rad) - 1.f).matrix().norm() < 1.f) { nK++; }
        } else {
          dimLoop(ii - 1, ind);
        }
      }
    };
    ind.setZero();
    dimLoop(NK - 1, ind);
  }
  oshape[0] = nK;
  Log::Print("Hankel", "ishape {} oshape {} kDims {} kW {}", ishape, oshape, kDims_, kW_);
}

template <int ND, int NK> void Hankel<ND, NK>::forward(InCMap x, OutMap y) const
{
  auto const             time = this->startForward(x, y, false);
  Index                  ik = 0;
  Sz<ND>                 st, roll, stSym, szSym;
  Eigen::array<bool, ND> rev;
  st.fill(0);
  roll.fill(0);
  stSym.fill(0);
  rev.fill(false);
  szSym = ishape;
  Eigen::Array<float, NK, 1> rad, ind;
  for (Index ii = 0; ii < NK; ii++) {
    Index const D = kDims_[ii];
    stSym[D] = 1;
    szSym[D] -= 1;
    rev[D] = true;
    if (sphere_) { rad[ii] = ishape[D] / 2; }
  }

  std::function<void(Index, Eigen::Array<float, NK, 1>)> dimLoop = [&](Index ii, Eigen::Array<float, NK, 1> ind) {
    Index const D = kDims_[ii];
    Index const nKd = x.dimension(D) - 1; // -1 because we are dropping the -N/2 samples
    for (Index id = 0; id < nKd; id++) {
      roll[D] = id;
      ind[ii] = id;
      if (ii == 0) {
        // rl::Log::Print("{} {} {} {}", id, ind, rad, ((ind / rad) - 1.f).matrix().norm());
        if (!sphere_ || ((ind / rad) - 1.f).matrix().norm() < 1.f) {
          y.template chip<1>(0).template chip<0>(ik) = x.slice(stSym, szSym).roll(roll).slice(st, kSz_);
          if (virt_) {
            y.template chip<1>(1).template chip<0>(ik) =
              x.slice(stSym, szSym).reverse(rev).roll(roll).slice(st, kSz_).conjugate();
          }
          ik++;
        }
      } else {
        dimLoop(ii - 1, ind);
      }
    }
  };
  ind.setZero();
  dimLoop(NK - 1, ind);
  assert(ik == y.dimension(0));
  this->finishForward(y, time, false);
}

template <int ND, int NK> void Hankel<ND, NK>::adjoint(OutCMap y, InMap x) const
{
  auto const time = this->startAdjoint(y, x, false);
  x.setZero();
  Index                  ik = 0;
  Sz<ND>                 xSt, roll, stSym, szSym;
  Eigen::array<bool, ND> rev;
  xSt.fill(0);
  roll.fill(0);
  stSym.fill(0);
  rev.fill(false);
  szSym = ishape;
  Eigen::Array<float, NK, 1> rad, ind;
  for (Index ii = 0; ii < NK; ii++) {
    Index const D = kDims_[ii];
    stSym[D] = 1;
    szSym[D] -= 1;
    rev[D] = true;
    if (sphere_) { rad[ii] = ishape[D] / 2; }
  }

  std::function<void(Index, Eigen::Array<float, NK, 1>)> dimLoop = [&](Index ii, Eigen::Array<float, NK, 1> ind) {
    Index const D = kDims_[ii];
    Index const nKd = x.dimension(D) - 1;
    for (Index id = 0; id < nKd; id++) {
      roll[D] = id;
      ind[ii] = id;
      if (ii == 0) {
        if (ii == 0) {
          if (!sphere_ || ((ind / rad) - 1.f).matrix().norm() < 1.f) {
            x.slice(stSym, szSym).roll(roll).slice(xSt, kSz_) += y.template chip<1>(0).template chip<0>(ik);
            if (virt_) {
              x.slice(stSym, szSym).reverse(rev).roll(roll).slice(xSt, kSz_) +=
                y.template chip<1>(1).template chip<0>(ik).conjugate();
            }
            ik++;
          }
        } else {
          dimLoop(ii - 1, ind);
        }
      }
    }
  };
  ind.setZero();
  dimLoop(NK - 1, ind);
  assert(ik == y.dimension(0));
  x.slice(stSym, szSym) /= x.slice(stSym, szSym).constant(Product(kW_) * (virt_ ? 2.f : 1.f));
  this->finishAdjoint(x, time, false);
}

template struct Hankel<5, 3>;
template struct Hankel<5, 1>;

} // namespace rl::TOps
