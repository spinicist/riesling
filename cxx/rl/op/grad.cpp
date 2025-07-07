#include "grad.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "top-impl.hpp"

namespace rl::TOps {

namespace {
template <bool fwd, typename T1, typename T2, typename SzT>
inline auto ForwardDiff(T1 const &a, T2 &&b, SzT const dims, Index const dim, float const s = 1.f)
{
  auto         sz = dims;
  decltype(sz) f0, fp1;
  fp1[dim] = 1;
  sz[dim] -= 1;
  if constexpr (fwd) {
    Log::Debug("Grad", "Forward forward differences dim {}", dim, dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(fp1, sz) - a.slice(f0, sz)) * b.slice(f0, sz).constant(s);
  } else {
    Log::Debug("Grad", "Adjoint forward differences dim {}", dim, dim);
    b.slice(fp1, sz).device(Threads::TensorDevice()) += (a.slice(f0, sz) - a.slice(fp1, sz)) * b.slice(f0, sz).constant(s);
  }
}

template <bool fwd, typename T1, typename T2, typename SzT>
inline auto BackwardDiff(T1 const &a, T2 &&b, SzT const dims, Index const dim, float const s = 1.f)
{
  auto         sz = dims;
  decltype(sz) f0, fp1;
  fp1[dim] = 1;
  sz[dim] -= 1;
  if constexpr (fwd) {
    Log::Debug("Grad", "Forward backward differences dim {}", dim);
    b.slice(fp1, sz).device(Threads::TensorDevice()) += (a.slice(fp1, sz) - a.slice(f0, sz)) * b.slice(f0, sz).constant(s);
  } else {
    Log::Debug("Grad", "Adjoint backward differences dim {}", dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) += (a.slice(f0, sz) - a.slice(fp1, sz)) * b.slice(f0, sz).constant(s);
  }
}

template <bool fwd, typename T1, typename T2, typename SzT>
inline auto CentralDiff0(T1 const &a, T2 &&b, SzT const dims, Index const dim, float const s = 1.f)
{
  auto         sz = dims;
  decltype(sz) fm1, f0, fp1;
  fp1[dim] = 2;
  fm1[dim] = 0;
  f0[dim] = 1;
  sz[dim] -= 2;
  if constexpr (fwd) {
    Log::Debug("Grad", "Forward central differences dim {}", dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) +=
      (a.slice(fp1, sz) - a.slice(fm1, sz)) * b.slice(f0, sz).constant(s / 2.f);
  } else {
    Log::Debug("Grad", "Adjoint central differences dim {}", dim);
    b.slice(f0, sz).device(Threads::TensorDevice()) +=
      (a.slice(fm1, sz) - a.slice(fp1, sz)) * b.slice(f0, sz).constant(s / 2.f);
  }
}

template <bool fwd, typename T1, typename T2, typename SzT>
inline auto CentralDiff1(T1 const &a, T2 &&b, SzT const dims, Index const dim, float const s = 1.f)
{ // Thanks http://www.holoborodko.com/pavel/numerical-methods/numerical-derivative/central-differences/
  auto         sz = dims;
  decltype(sz) f0, fp1, fp2, fm1, fm2;
  f0[dim] = 2;
  fp1[dim] = 3;
  fp2[dim] = 4;
  fm1[dim] = 1;
  fm2[dim] = 0;
  sz[dim] -= 4;
  if constexpr (fwd) {
    b.slice(f0, sz).device(Threads::TensorDevice()) +=
      (a.slice(fm2, sz) - 8.f * a.slice(fm1, sz) + 8.f * a.slice(fp1, sz) - a.slice(fp2, sz)) *
      b.slice(f0, sz).constant(s / 12.f);
  } else {
    b.slice(f0, sz).device(Threads::TensorDevice()) +=
      (a.slice(fp2, sz) - 8.f * a.slice(fp1, sz) + 8.f * a.slice(fm1, sz) - a.slice(fm2, sz)) *
      b.slice(f0, sz).constant(s / 12.f);
  }
}
} // namespace

template <int ND, int NG> Grad<ND, NG>::Grad(InDims const ish, Sz<NG> const d, int const o)
  : Parent("Grad", ish, AddBack(ish, (Index)d.size()))
  , dims_{d}
  , mode_{o}
{
  if (mode_ < 0 || mode_ > 3) { throw(Log::Failure("Grad", "Invalid gradient mode {}", mode_)); }
}

template <int ND, int NG> auto Grad<ND, NG>::Make(InDims const ish, Sz<NG> const d, int const o) -> std::shared_ptr<Grad>
{
  return std::make_shared<Grad>(ish, d, o);
}

template <int ND, int NG> void Grad<ND, NG>::forward(InCMap x, OutMap y) const
{
  y.setZero();
  iforward(x, y);
}

template <int ND, int NG> void Grad<ND, NG>::adjoint(OutCMap y, InMap x) const
{
  x.setZero();
  iadjoint(y, x);
}

template <int ND, int NG> void Grad<ND, NG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const  time = this->startForward(x, y, true);
  float const scale = s / std::sqrt(2 * NG);
  for (Index ii = 0; ii < NG; ii++) {
    switch (mode_) {
    case 0: ForwardDiff<true>(x, y.template chip<ND>(ii), x.dimensions(), dims_[ii], scale); break;
    case 1: BackwardDiff<true>(x, y.template chip<ND>(ii), x.dimensions(), dims_[ii], scale); break;
    case 2: CentralDiff0<true>(x, y.template chip<ND>(ii), x.dimensions(), dims_[ii], scale); break;
    case 3: CentralDiff1<true>(x, y.template chip<ND>(ii), x.dimensions(), dims_[ii], scale); break;
    }
  }
  this->finishForward(y, time, true);
}

template <int ND, int NG> void Grad<ND, NG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const  time = this->startAdjoint(y, x, true);
  float const scale = s / std::sqrt(2 * NG);
  for (Index ii = 0; ii < NG; ii++) {
    switch (mode_) {
    case 0: ForwardDiff<false>(y.template chip<ND>(ii), x, x.dimensions(), dims_[ii], scale); break;
    case 1: BackwardDiff<false>(y.template chip<ND>(ii), x, x.dimensions(), dims_[ii], scale); break;
    case 2: CentralDiff0<false>(y.template chip<ND>(ii), x, x.dimensions(), dims_[ii], scale); break;
    case 3: CentralDiff1<false>(y.template chip<ND>(ii), x, x.dimensions(), dims_[ii], scale); break;
    }
  }
  this->finishAdjoint(x, time, true);
}

template struct Grad<5, 1>;
template struct Grad<5, 3>;

template <int ND, int NG> Div<ND, NG>::Div(OutDims const sh, Sz<NG> const d, int const o)
  : Parent("Grad", AddBack(sh, (Index)d.size()), sh)
  , dims_{d}
  , mode_{o}
{
  if (mode_ < 0 || mode_ > 3) { throw(Log::Failure("Grad", "Invalid gradient mode {}", mode_)); }
}

template <int ND, int NG> auto Div<ND, NG>::Make(OutDims const sh, Sz<NG> const d, int const o) -> std::shared_ptr<Div>
{
  return std::make_shared<Div>(sh, d, o);
}

template <int ND, int NG> void Div<ND, NG>::forward(InCMap x, OutMap y) const
{
  y.setZero();
  iforward(x, y);
}

template <int ND, int NG> void Div<ND, NG>::adjoint(OutCMap y, InMap x) const
{
  x.setZero();
  iadjoint(y, x);
}

template <int ND, int NG> void Div<ND, NG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);

  float const scale = s / std::sqrt(2 * NG);
  for (Index ii = 0; ii < NG; ii++) {
    switch (mode_) {
    case 0: BackwardDiff<true>(x.template chip<ND>(ii), y, y.dimensions(), dims_[ii], scale); break;
    case 1: ForwardDiff<true>(x.template chip<ND>(ii), y, y.dimensions(), dims_[ii], scale); break;
    case 2: CentralDiff0<true>(x.template chip<ND>(ii), y, y.dimensions(), dims_[ii], scale); break;
    case 3: CentralDiff1<true>(x.template chip<ND>(ii), y, y.dimensions(), dims_[ii], scale); break;
    }
  }
  this->finishForward(y, time, true);
}

template <int ND, int NG> void Div<ND, NG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const  time = this->startAdjoint(y, x, true);
  float const scale = s / std::sqrt(2 * NG);
  for (Index ii = 0; ii < NG; ii++) {
    switch (mode_) {
    case 0: BackwardDiff<false>(y, x.template chip<ND>(ii), y.dimensions(), dims_[ii], scale); break;
    case 1: ForwardDiff<false>(y, x.template chip<ND>(ii), y.dimensions(), dims_[ii], scale); break;
    case 2: CentralDiff0<false>(y, x.template chip<ND>(ii), y.dimensions(), dims_[ii], scale); break;
    case 3: CentralDiff1<false>(y, x.template chip<ND>(ii), y.dimensions(), dims_[ii], scale); break;
    }
  }
  this->finishAdjoint(x, time, true);
}

template struct Div<5, 3>;

template <int ND, int NG> GradVec<ND, NG>::GradVec(InDims const ishape, Sz<NG> const dims, int const o)
  : Parent("GradV", ishape, AddBack(FirstN<ND - 1>(ishape), (Index)((dims.size() * (dims.size() + 1)) / 2)))
  , dims_{dims}
  , mode_{o}
{
  if ((Index)dims.size() != ishape[ND - 1]) {
    throw(Log::Failure("gradv", "Symmetrized gradient only, dims were {} and {}", dims.size(), ishape[ND - 1]));
  }
  if (mode_ < 0 || mode_ > 3) { throw(Log::Failure("Grad", "Invalid gradient mode {}", mode_)); }
}

template <int ND, int NG> auto GradVec<ND, NG>::Make(InDims const ish, Sz<NG> const d, int const o) -> std::shared_ptr<GradVec>
{
  return std::make_shared<GradVec>(ish, d, o);
}

template <int ND, int NG> void GradVec<ND, NG>::forward(InCMap x, OutMap y) const
{
  y.setZero();
  iforward(x, y);
}

template <int ND, int NG> void GradVec<ND, NG>::iforward(InCMap x, OutMap y, float const s) const
{
  auto const time = this->startForward(x, y, true);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  /*
   * Grad applied to a vector produces a tensor. Here it is flattened back into a vector
   */
  float const scale = s / std::sqrt(2 * NG * (NG + 1) / 2);
  Index yind = NG;
  for (Index ii = 0; ii < NG; ii++) {
    switch (mode_) {
    case 0: BackwardDiff<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(ii), sz, dims_[ii], scale); break;
    case 1: ForwardDiff<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(ii), sz, dims_[ii], scale); break;
    case 2: CentralDiff0<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(ii), sz, dims_[ii], scale); break;
    case 3: CentralDiff1<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(ii), sz, dims_[ii], scale); break;
    }
    for (Index ij = ii + 1; ij < NG; ij++) {
      switch (mode_) {
      case 0:
        BackwardDiff<true>(x.template chip<ND - 1>(ij), y.template chip<ND - 1>(yind), sz, dims_[ii], scale / std::sqrt(2));
        BackwardDiff<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(yind), sz, dims_[ij], scale / std::sqrt(2));
        break;
      case 1:
        ForwardDiff<true>(x.template chip<ND - 1>(ij), y.template chip<ND - 1>(yind), sz, dims_[ii], scale / std::sqrt(2));
        ForwardDiff<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(yind), sz, dims_[ij], scale / std::sqrt(2));
        break;
      case 2:
        CentralDiff0<true>(x.template chip<ND - 1>(ij), y.template chip<ND - 1>(yind), sz, dims_[ii], scale / std::sqrt(2));
        CentralDiff0<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(yind), sz, dims_[ij], scale / std::sqrt(2));
        break;
      case 3:
        CentralDiff1<true>(x.template chip<ND - 1>(ij), y.template chip<ND - 1>(yind), sz, dims_[ii], scale / std::sqrt(2));
        CentralDiff1<true>(x.template chip<ND - 1>(ii), y.template chip<ND - 1>(yind), sz, dims_[ij], scale / std::sqrt(2));
        break;
      }
      yind++;
    }
  }
  this->finishForward(y, time, false);
}

template <int ND, int NG> void GradVec<ND, NG>::adjoint(OutCMap y, InMap x) const
{
  x.setZero();
  iadjoint(y, x);
}

template <int ND, int NG> void GradVec<ND, NG>::iadjoint(OutCMap y, InMap x, float const s) const
{
  auto const time = this->startAdjoint(y, x, true);
  auto const sz = FirstN<ND - 1>(x.dimensions());
  /*
   *  This is the tensor form of Div (see wikipedia page) but with the tensor flattened into a vector
   */
  float const scale = s / std::sqrt(2 * NG * (NG + 1) / 2);
  Index yind = NG;
  for (Index ii = 0; ii < NG; ii++) {
    switch (mode_) {
    case 0: BackwardDiff<false>(y.template chip<ND - 1>(ii), x.template chip<ND - 1>(ii), sz, dims_[ii], scale); break;
    case 1: ForwardDiff<false>(y.template chip<ND - 1>(ii), x.template chip<ND - 1>(ii), sz, dims_[ii], scale); break;
    case 2: CentralDiff0<false>(y.template chip<ND - 1>(ii), x.template chip<ND - 1>(ii), sz, dims_[ii], scale); break;
    case 3: CentralDiff1<false>(y.template chip<ND - 1>(ii), x.template chip<ND - 1>(ii), sz, dims_[ii], scale); break;
    }
    for (Index ij = ii + 1; ij < NG; ij++) {
      switch (mode_) {
      case 0:
        BackwardDiff<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ii), sz, dims_[ij], scale / std::sqrt(2));
        BackwardDiff<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ij), sz, dims_[ii], scale / std::sqrt(2));
        break;
      case 1:
        ForwardDiff<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ii), sz, dims_[ij], scale / std::sqrt(2));
        ForwardDiff<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ij), sz, dims_[ii], scale / std::sqrt(2));
        break;
      case 2:
        CentralDiff0<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ii), sz, dims_[ij], scale / std::sqrt(2));
        CentralDiff0<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ij), sz, dims_[ii], scale / std::sqrt(2));
        break;
      case 3:
        CentralDiff1<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ii), sz, dims_[ij], scale / std::sqrt(2));
        CentralDiff1<false>(y.template chip<ND - 1>(yind), x.template chip<ND - 1>(ij), sz, dims_[ii], scale / std::sqrt(2));
        break;
      }
      yind++;
    }
  }
  this->finishAdjoint(x, time, false);
}

template struct GradVec<6, 3>;

} // namespace rl::TOps
